#!/usr/bin/env python3
"""
SURVIVOR_SCORER.PY PATCH - Step 6 Restoration v2.2
===================================================

Apply these changes to survivor_scorer.py:

1. CHANGE LINE 116:
   FROM: self.generate_sequence = java_lcg_sequence
   TO:   self._cpu_func = get_cpu_reference(self.prng_type)

2. ADD these methods to the SurvivorScorer class

3. UPDATE LINE 124 in extract_ml_features():
   FROM: seq = self.generate_sequence(seed, n, self.mod)
   TO:   seq = self._generate_sequence(seed, n, skip=skip)
"""

# =============================================================================
# METHOD 1: Add after __init__ (around line 118)
# =============================================================================

def _generate_sequence(self, seed: int, n: int, skip: int = 0) -> np.ndarray:
    """
    Generate PRNG sequence using configured prng_type.
    
    Uses prng_registry for dynamic PRNG lookup - NO HARDCODING.
    
    Args:
        seed: PRNG seed value
        n: Number of values to generate
        skip: Skip value for PRNG (default 0)
        
    Returns:
        np.ndarray of generated values with mod applied
    """
    raw = self._cpu_func(seed=int(seed), n=n, skip=skip)
    return np.array([v % self.mod for v in raw], dtype=np.int64)


# =============================================================================
# METHOD 2: Add after _generate_sequence
# =============================================================================

def compute_dual_sieve_intersection(
    self,
    forward_survivors: List[int],
    reverse_survivors: List[int]
) -> Dict[str, Any]:
    """
    Compute intersection of forward and reverse sieve survivors.
    
    Per Team Beta v2.2 Requirements:
    - NEVER discard valid intersection (even with low Jaccard)
    - Return intersection as sorted list for determinism
    - Jaccard is metadata for confidence weighting, not a filter
    
    Args:
        forward_survivors: Seeds from forward sieve
        reverse_survivors: Seeds from reverse sieve
        
    Returns:
        Dict with:
        - intersection: Sorted list of bidirectional survivors
        - jaccard: Jaccard similarity index (|A∩B| / |A∪B|)
        - counts: Dict with forward, reverse, intersection, union counts
    """
    # Handle empty inputs
    if not forward_survivors or not reverse_survivors:
        return {
            "intersection": [],
            "jaccard": 0.0,
            "counts": {
                "forward": len(forward_survivors) if forward_survivors else 0,
                "reverse": len(reverse_survivors) if reverse_survivors else 0,
                "intersection": 0,
                "union": 0
            }
        }
    
    forward_set = set(forward_survivors)
    reverse_set = set(reverse_survivors)
    
    intersection = forward_set & reverse_set
    union = forward_set | reverse_set
    
    # Jaccard similarity: |A ∩ B| / |A ∪ B|
    jaccard = len(intersection) / len(union) if union else 0.0
    
    # ALWAYS return intersection - never discard valid data
    # Sort for deterministic behavior
    return {
        "intersection": sorted(list(intersection)),
        "jaccard": float(jaccard),
        "counts": {
            "forward": len(forward_survivors),
            "reverse": len(reverse_survivors),
            "intersection": len(intersection),
            "union": len(union)
        }
    }


# =============================================================================
# FULL PATCH INSTRUCTIONS
# =============================================================================
"""
To apply this patch:

1. Open survivor_scorer.py

2. Find line ~116:
   self.generate_sequence = java_lcg_sequence
   
   REPLACE WITH:
   self._cpu_func = get_cpu_reference(self.prng_type)

3. Find the extract_ml_features method (~line 119)
   After the __init__ method ends (after self.device = ...), ADD:
   
   [paste _generate_sequence method here]
   [paste compute_dual_sieve_intersection method here]

4. In extract_ml_features method, find line ~124:
   seq = self.generate_sequence(seed, n, self.mod)
   
   REPLACE WITH:
   seq = self._generate_sequence(seed, n, skip=skip)

5. Test:
   python3 -c "
   from survivor_scorer import SurvivorScorer
   scorer = SurvivorScorer(prng_type='java_lcg', mod=1000)
   
   # Test PRNG
   seq = scorer._generate_sequence(12345, 10)
   print(f'Sequence: {seq[:5]}')
   
   # Test intersection
   result = scorer.compute_dual_sieve_intersection([1,2,3,4], [3,4,5,6])
   print(f'Intersection: {result}')
   assert result['intersection'] == [3, 4]
   print('✅ survivor_scorer patch successful')
   "
"""
