#!/usr/bin/env python3
"""PRNG Sweep Orchestrator - Uses Python API"""
import json, time, sys
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import argparse
from dataclasses import dataclass, asdict

sys.path.insert(0, '/home/michael/distributed_prng_analysis')
from prng_registry import list_available_prngs

@dataclass
class PRNGTestResult:
    prng_name: str
    prng_type: str = 'base'
    optimal_window_size: Optional[int] = None
    optimal_offset: Optional[int] = None
    forward_survivors: Optional[int] = None
    reverse_survivors: Optional[int] = None
    bidirectional_survivors: Optional[int] = None
    intersection_quality: Optional[float] = None
    timestamp: str = ""
    duration_total: Optional[float] = None
    status: str = "pending"
    error: Optional[str] = None

class PRNGSweepOrchestrator:
    def __init__(self, target_file, output_dir="prng_sweep_results", 
                 session=None, threshold=0.01, skip_window_opt=False):
        self.target_file, self.output_dir = target_file, output_dir
        self.session, self.threshold = session, threshold
        self.skip_window_opt = skip_window_opt
        Path(output_dir).mkdir(exist_ok=True)
        self.results_dir = f"{output_dir}/individual_results"
        Path(self.results_dir).mkdir(exist_ok=True)
        self.all_prngs = list_available_prngs()
        self.base_prngs = [p for p in self.all_prngs 
                          if not any(x in p for x in ['_hybrid', '_reverse'])]
        print(f"✅ {len(self.all_prngs)} PRNGs ({len(self.base_prngs)} base)")
    
    def test_single_prng(self, prng_name):
        print(f"\n{'='*70}\n🧪 {prng_name}\n{'='*70}")
        result = PRNGTestResult(prng_name=prng_name, timestamp=datetime.now().isoformat())
        start = time.time()
        try:
            from coordinator import MultiGPUCoordinator
            from window_optimizer_integration_final import add_window_optimizer_to_coordinator, run_bidirectional_test
            from window_optimizer import WindowConfig
            
            window_size, offset = 512, 0
            if not self.skip_window_opt:
                print("  🔍 Window optimization...")
                add_window_optimizer_to_coordinator()
                coord = MultiGPUCoordinator('distributed_config.json')
                opt_res = coord.optimize_window(
                    dataset_path=self.target_file, seed_start=0, seed_count=10_000_000,
                    prng_base=prng_name, strategy_name='bayesian', max_iterations=5,
                    output_file=f'{self.results_dir}/{prng_name}_wopt.json')
                window_size = opt_res['best_config']['window_size']
                offset = opt_res['best_config']['offset']
                print(f"  ✅ window={window_size}, offset={offset}")
            
            result.optimal_window_size, result.optimal_offset = window_size, offset
            print("  🔄 Bidirectional test...")
            coord = MultiGPUCoordinator('distributed_config.json')
            config = WindowConfig(window_size=window_size, offset=offset,
                sessions=['midday','evening'] if not self.session else [self.session],
                skip_min=0, skip_max=30)
            
            test_res = run_bidirectional_test(coord, config, self.target_file,
                0, 10_000_000, prng_name, self.threshold)
            
            result.forward_survivors = test_res.forward_count
            result.reverse_survivors = test_res.reverse_count
            result.bidirectional_survivors = test_res.bidirectional_count
            result.intersection_quality = (test_res.bidirectional_count / test_res.forward_count 
                                          if test_res.forward_count > 0 else 0)
            
            print(f"  📊 Fwd:{test_res.forward_count:,} Rev:{test_res.reverse_count:,} Bi:{test_res.bidirectional_count:,}")
            result.duration_total = time.time() - start
            result.status = "completed"
        except Exception as e:
            result.status, result.error = "failed", str(e)
            print(f"  ❌ {e}")
        return result
    
    def run_full_sweep(self, prng_list=None):
        prng_list = prng_list or self.base_prngs
        print(f"\n{'='*70}\n🚀 SWEEP: {len(prng_list)} PRNGs\n{'='*70}")
        results = {}
        start = time.time()
        for i, prng in enumerate(prng_list, 1):
            print(f"\n[{i}/{len(prng_list)}]")
            results[prng] = self.test_single_prng(prng)
            with open(f'{self.results_dir}/{prng}_results.json', 'w') as f:
                json.dump(asdict(results[prng]), f, indent=2)
        self.generate_report(results, time.time() - start)
        return results
    
    def generate_report(self, results, total_time):
        summary = {'sweep_metadata': {'timestamp': datetime.now().isoformat(),
            'target_file': self.target_file, 'total_prngs_tested': len(results),
            'total_duration_seconds': total_time},
            'prng_results': {n: asdict(r) for n, r in results.items()}}
        with open(f'{self.output_dir}/sweep_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        rankings = [(n, r.bidirectional_survivors or 0, r.intersection_quality or 0)
                   for n, r in results.items() if r.status == "completed"]
        rankings.sort(key=lambda x: (x[1] if x[1] > 0 else 999999, -x[2]))
        
        with open(f'{self.output_dir}/prng_rankings.txt', 'w') as f:
            f.write("="*70 + "\nPRNG RANKINGS\n" + "="*70 + "\n\n")
            for i, (p, c, q) in enumerate(rankings, 1):
                f.write(f"#{i}: {p}\n  Bi:{c:,} Q:{q*100:.1f}%\n\n")
        
        print(f"\n{'='*70}\n📊 DONE! {total_time/60:.1f}min\n{'='*70}")
        for i, (p, c, q) in enumerate(rankings[:5], 1):
            print(f"{i}. {p:20} {c:,} ({q*100:.1f}%)")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', required=True)
    p.add_argument('--prngs', nargs='*')
    p.add_argument('--session', choices=['midday','evening'])
    p.add_argument('--threshold', type=float, default=0.01)
    p.add_argument('--output-dir', default='prng_sweep_results')
    p.add_argument('--skip-window-opt', action='store_true')
    args = p.parse_args()
    PRNGSweepOrchestrator(args.target, args.output_dir, args.session,
        args.threshold, args.skip_window_opt).run_full_sweep(args.prngs)

if __name__ == '__main__':
    main()
