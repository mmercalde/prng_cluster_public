#!/usr/bin/env python3
"""
Automated State Reconstruction Attack on Number 390 Occurrences
Compares PRNG states across the 3.5-year gap
"""

import json
import sys
from advanced_search_manager import AdvancedSearchManager, StateReconstructionConfig

def run_reconstruction_attack():
    # Load target sequences
    with open('target_sequences_390.json', 'r') as f:
        targets = json.load(f)
    
    manager = AdvancedSearchManager()
    results = {}
    
    print("=" * 70)
    print("AUTOMATED STATE RECONSTRUCTION ATTACK")
    print("=" * 70)
    print()
    
    # Attack each sequence
    for name, target_data in targets.items():
        sequence = target_data['sequence']
        description = target_data['description']
        
        print(f"\n{'='*70}")
        print(f"ATTACKING: {description}")
        print(f"{'='*70}")
        print(f"Sequence length: {len(sequence)}")
        print(f"Target value 390 at position: {sequence.index(390)}")
        print()
        
        # Try MT19937 with bruteforce (CORRECT METHOD NAME)
        print("Testing MT19937 (bruteforce)...")
        config_mt_brute = StateReconstructionConfig(
            prng_type='mt',  # Changed from 'mt19937' to 'mt'
            known_sequence=sequence,
            sequence_length=len(sequence),
            reconstruction_method='bruteforce',
            priority=1
        )
        
        try:
            search_id = manager.create_state_reconstruction(config_mt_brute)
            print(f"  ✅ MT19937 bruteforce search started: {search_id}")
            results[f"{name}_mt_bruteforce"] = search_id
        except Exception as e:
            print(f"  ❌ MT19937 bruteforce failed: {e}")
        
        # Try MT19937 with algebraic (faster method)
        print("Testing MT19937 (algebraic)...")
        config_mt_alg = StateReconstructionConfig(
            prng_type='mt',
            known_sequence=sequence,
            sequence_length=len(sequence),
            reconstruction_method='algebraic',
            priority=1
        )
        
        try:
            search_id = manager.create_state_reconstruction(config_mt_alg)
            print(f"  ✅ MT19937 algebraic search started: {search_id}")
            results[f"{name}_mt_algebraic"] = search_id
        except Exception as e:
            print(f"  ❌ MT19937 algebraic failed: {e}")
        
        # Try LCG with bruteforce
        print("Testing LCG (bruteforce)...")
        config_lcg = StateReconstructionConfig(
            prng_type='lcg',
            known_sequence=sequence,
            sequence_length=len(sequence),
            reconstruction_method='bruteforce',
            priority=2
        )
        
        try:
            search_id = manager.create_state_reconstruction(config_lcg)
            print(f"  ✅ LCG bruteforce search started: {search_id}")
            results[f"{name}_lcg_bruteforce"] = search_id
        except Exception as e:
            print(f"  ❌ LCG bruteforce failed: {e}")
    
    # Save results
    with open('attack_results_390.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("=" * 70)
    print("ATTACK LAUNCHED!")
    print("=" * 70)
    print(f"Total searches initiated: {len(results)}")
    print("Search IDs saved to: attack_results_390.json")
    print()
    print("Monitor progress with:")
    print("  python3 unified_system_working.py")
    print("  → Option 3: Advanced Research")
    print("  → Option 7: View Search Progress")
    print()
    
    return results

if __name__ == "__main__":
    try:
        results = run_reconstruction_attack()
        
        print("\n" + "=" * 70)
        print("SEARCH IDs:")
        print("=" * 70)
        for name, search_id in results.items():
            print(f"  {name}: {search_id}")
    except Exception as e:
        print(f"\n❌ Attack failed: {e}")
        import traceback
        traceback.print_exc()

