#!/usr/bin/env python3
"""
Test Reinforcement Engine with Real Forward/Reverse Survivors
==============================================================

This script demonstrates how to integrate the reinforcement engine
with your existing forward/reverse sieve pipeline for ML/AI automation.

Pipeline Integration Flow:
1. Load lottery history (daily3.json)
2. Load forward survivors (from forward sieve)
3. Load reverse survivors (from reverse sieve)
4. Initialize reinforcement engine
5. Score all survivors with ML
6. Train on historical performance (if available)
7. Rank and prune survivor pool
8. Save results for prediction generation

Author: Distributed PRNG Analysis System
Date: November 6, 2025
"""

from reinforcement_engine import ReinforcementEngine, ReinforcementConfig
import json
import numpy as np
from pathlib import Path

def load_lottery_history(filename='daily3.json'):
    """Load lottery history from JSON file"""
    print(f"📂 Loading lottery history from {filename}...")
    with open(filename) as f:
        lottery_data = json.load(f)
    
    # Extract just the draw numbers
    lottery_history = [d['draw'] for d in lottery_data]
    print(f"   ✅ Loaded {len(lottery_history)} draws")
    return lottery_history, lottery_data

def load_survivors(filename):
    """Load survivor seeds from JSON file"""
    print(f"📂 Loading survivors from {filename}...")
    with open(filename) as f:
        survivors = json.load(f)
    print(f"   ✅ Loaded {len(survivors)} survivors")
    return survivors

def score_survivors(engine, survivors, label="Survivors"):
    """Score a list of survivors using the reinforcement engine"""
    print(f"\n🔍 Scoring {len(survivors)} {label}...")
    qualities = engine.predict_quality_batch(survivors)
    
    # Get statistics
    mean_quality = np.mean(qualities)
    std_quality = np.std(qualities)
    min_quality = np.min(qualities)
    max_quality = np.max(qualities)
    
    print(f"   Quality Stats:")
    print(f"     Mean: {mean_quality:.6f}")
    print(f"     Std:  {std_quality:.6f}")
    print(f"     Min:  {min_quality:.6f}")
    print(f"     Max:  {max_quality:.6f}")
    
    return qualities

def show_top_survivors(survivors, qualities, top_n=10, label="Top Survivors"):
    """Display top N survivors by quality"""
    print(f"\n🏆 {label} (Top {top_n}):")
    top_indices = np.argsort(qualities)[-top_n:][::-1]
    
    for rank, idx in enumerate(top_indices, 1):
        seed = survivors[idx]
        quality = qualities[idx]
        print(f"   {rank:2d}. Seed {seed:8d}: quality={quality:.6f}")
    
    return [survivors[i] for i in top_indices]

def compare_forward_reverse(forward_survivors, forward_qualities, 
                           reverse_survivors, reverse_qualities):
    """Compare forward vs reverse survivor performance"""
    print(f"\n📊 Forward vs Reverse Comparison:")
    
    fwd_mean = np.mean(forward_qualities)
    rev_mean = np.mean(reverse_qualities)
    
    print(f"   Forward mean quality: {fwd_mean:.6f}")
    print(f"   Reverse mean quality: {rev_mean:.6f}")
    
    if fwd_mean > rev_mean:
        diff = ((fwd_mean - rev_mean) / rev_mean) * 100
        print(f"   ✅ Forward is {diff:.2f}% better")
    else:
        diff = ((rev_mean - fwd_mean) / fwd_mean) * 100
        print(f"   ✅ Reverse is {diff:.2f}% better")

def train_on_historical_performance(engine, survivors, lottery_history):
    """
    Simulate training on historical performance
    
    In production, you would:
    1. Run survivors against historical data
    2. Calculate actual hit rates
    3. Train the model on actual performance
    
    For this demo, we'll simulate some performance data
    """
    print(f"\n📚 Training on simulated historical performance...")
    
    # Simulate hit rates (in production, use real hit@10 rates)
    # Higher quality seeds would have higher hit rates
    simulated_results = np.random.uniform(0.4, 0.8, len(survivors))
    
    print(f"   Training samples: {len(survivors)}")
    print(f"   Result range: [{np.min(simulated_results):.2f}, {np.max(simulated_results):.2f}]")
    
    # Train the model
    engine.train(survivors, simulated_results.tolist())
    print(f"   ✅ Training complete")

def prune_survivor_pool(engine, all_survivors, keep_top_n=100):
    """
    Prune survivor pool to top performers
    
    This is the key ML integration step - the engine learns
    which survivors perform best and keeps only the top ones.
    """
    print(f"\n✂️  Pruning survivor pool...")
    print(f"   Current pool size: {len(all_survivors)}")
    print(f"   Target size: {keep_top_n}")
    
    top_survivors = engine.prune_survivors(all_survivors, keep_top_n=keep_top_n)
    
    print(f"   ✅ Pruned to {len(top_survivors)} survivors")
    return top_survivors

def save_results(top_forward, top_reverse, output_file='ml_ranked_survivors.json'):
    """Save ML-ranked survivors for use in prediction generation"""
    print(f"\n💾 Saving ML-ranked survivors to {output_file}...")
    
    results = {
        'timestamp': str(np.datetime64('now')),
        'top_forward_survivors': top_forward,
        'top_reverse_survivors': top_reverse,
        'total_survivors': len(top_forward) + len(top_reverse)
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ✅ Saved {results['total_survivors']} survivors")

def main():
    """
    Main pipeline integration test
    
    This demonstrates the complete ML/AI workflow:
    1. Load data
    2. Initialize ML engine
    3. Score survivors
    4. Train on performance
    5. Re-score and rank
    6. Prune pool
    7. Save for predictions
    """
    print("="*70)
    print("REINFORCEMENT ENGINE - REAL SURVIVOR INTEGRATION TEST")
    print("="*70)
    
    # Step 1: Load lottery history
    lottery_history, lottery_data = load_lottery_history('daily3.json')
    
    # Step 2: Load forward and reverse survivors
    forward_survivors = load_survivors('forward_survivors.json')
    reverse_survivors = load_survivors('reverse_survivors.json')
    
    # Step 3: Initialize reinforcement engine
    print("\n🚀 Initializing Reinforcement Engine...")
    config = ReinforcementConfig.from_json('reinforcement_engine_config.json')
    engine = ReinforcementEngine(config, lottery_history)
    print("   ✅ Engine initialized")
    
    # Step 4: Initial scoring (before training)
    print("\n" + "="*70)
    print("PHASE 1: INITIAL SCORING (UNTRAINED)")
    print("="*70)
    
    forward_qualities_initial = score_survivors(engine, forward_survivors, "Forward Survivors")
    reverse_qualities_initial = score_survivors(engine, reverse_survivors, "Reverse Survivors")
    
    show_top_survivors(forward_survivors, forward_qualities_initial, 
                      top_n=min(10, len(forward_survivors)), 
                      label="Forward Survivors (Untrained)")
    show_top_survivors(reverse_survivors, reverse_qualities_initial, 
                      top_n=min(10, len(reverse_survivors)), 
                      label="Reverse Survivors (Untrained)")
    
    compare_forward_reverse(forward_survivors, forward_qualities_initial,
                           reverse_survivors, reverse_qualities_initial)
    
    # Step 5: Train on historical performance
    print("\n" + "="*70)
    print("PHASE 2: TRAINING")
    print("="*70)
    
    # Combine all survivors for training
    all_survivors = forward_survivors + reverse_survivors
    train_on_historical_performance(engine, all_survivors, lottery_history)
    
    # Step 6: Re-score after training
    print("\n" + "="*70)
    print("PHASE 3: POST-TRAINING SCORING")
    print("="*70)
    
    forward_qualities_trained = score_survivors(engine, forward_survivors, "Forward Survivors")
    reverse_qualities_trained = score_survivors(engine, reverse_survivors, "Reverse Survivors")
    
    top_forward = show_top_survivors(forward_survivors, forward_qualities_trained, 
                                     top_n=min(10, len(forward_survivors)), 
                                     label="Forward Survivors (Trained)")
    top_reverse = show_top_survivors(reverse_survivors, reverse_qualities_trained, 
                                     top_n=min(10, len(reverse_survivors)), 
                                     label="Reverse Survivors (Trained)")
    
    compare_forward_reverse(forward_survivors, forward_qualities_trained,
                           reverse_survivors, reverse_qualities_trained)
    
    # Step 7: Show improvement from training
    print("\n" + "="*70)
    print("TRAINING IMPACT ANALYSIS")
    print("="*70)
    
    fwd_improve = np.mean(forward_qualities_trained) - np.mean(forward_qualities_initial)
    rev_improve = np.mean(reverse_qualities_trained) - np.mean(reverse_qualities_initial)
    
    print(f"\n📈 Quality Improvement:")
    print(f"   Forward: {fwd_improve:+.6f} ({fwd_improve/np.mean(forward_qualities_initial)*100:+.2f}%)")
    print(f"   Reverse: {rev_improve:+.6f} ({rev_improve/np.mean(reverse_qualities_initial)*100:+.2f}%)")
    
    # Step 8: Prune survivor pool (optional, if you have many survivors)
    if len(all_survivors) > 100:
        print("\n" + "="*70)
        print("PHASE 4: SURVIVOR POOL PRUNING")
        print("="*70)
        
        top_survivors = prune_survivor_pool(engine, all_survivors, keep_top_n=100)
    
    # Step 9: Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    save_results(top_forward, top_reverse)
    
    # Step 10: Save trained model
    print("\n💾 Saving trained model...")
    engine.save_model('reinforcement_model_trained.pth')
    print("   ✅ Model saved: reinforcement_model_trained.pth")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ INTEGRATION TEST COMPLETE")
    print("="*70)
    print(f"\nResults:")
    print(f"  - Processed {len(forward_survivors)} forward + {len(reverse_survivors)} reverse survivors")
    print(f"  - Trained model on {len(all_survivors)} samples")
    print(f"  - Top performers saved to: ml_ranked_survivors.json")
    print(f"  - Trained model saved to: reinforcement_model_trained.pth")
    print(f"\nNext Steps:")
    print(f"  1. Use top survivors for prediction generation")
    print(f"  2. Evaluate predictions against next draw")
    print(f"  3. Continue learning loop with new data")
    print(f"  4. Integrate into unified_system_working.py menu")

if __name__ == '__main__':
    main()
