import sys; sys.path.append(".")
from results_manager import create_readable_filename, analyze_result_status
#!/usr/bin/env python3
"""
Direct Analysis Module - Real cluster analysis with coordinator.py integration
"""

import os
import time
import json
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional

class DirectAnalysis:
    """Direct cluster analysis functionality"""

    def __init__(self, core):
        """Initialize with core system reference"""
        self.core = core
        self.module_name = "DirectAnalysis"

    def menu(self):
        """Direct analysis menu"""
        while True:
            self.core.clear_screen()
            self.core.print_header()
            print(f"\n{self.module_name.upper()}")
            print("-" * 35)
            print("STANDARD ANALYSIS MODES:")
            print("  1. Quick Test Analysis")
            print("  2. Standard Analysis")
            print("  3. Comprehensive Analysis")
            print("  4. Production Scale Analysis")

            print("\nSPECIALIZED OPERATIONS:")
            print("  5. Custom Multi-PRNG Analysis")
            print("  6. Single Job Distribution")
            print("  7. Draw Matching Analysis")
            print("  8. Residue Sieve Analysis")
            print("  9. System Connectivity Test")

            print("\nADVANCED OPTIONS:")
            print(" 10. Batch Job Execution")
            print(" 11. Performance Benchmark")
            print(" 12. Back to Main Menu")
            print("-" * 35)

            choice = input("Select option (1-12): ").strip()

            if choice == '1':
                self.quick_test_analysis()
            elif choice == '2':
                self.standard_analysis()
            elif choice == '3':
                self.comprehensive_analysis()
            elif choice == '4':
                self.production_analysis()
            elif choice == '5':
                self.custom_analysis()
            elif choice == '6':
                self.single_job_distribution()
            elif choice == '7':
                self.draw_matching_analysis()
            elif choice == '8':
                self.residue_sieve_analysis()      # NEW
            elif choice == '9':
                self.connectivity_test()
            elif choice == '10':
                self.batch_execution()
            elif choice == '11':
                self.performance_benchmark()
            elif choice == '12':
                break
            else:
                print("Invalid choice. Press Enter to continue...")
                input()

    def quick_test_analysis(self):
        """Execute quick test analysis with optimized parameters"""
        print("\nQuick Test Analysis")
        print("=" * 40)
        print("Purpose: Fast connectivity and basic correlation test")
        print("Parameters: 1K seeds, 1K samples, light correlation analysis")
        print("Estimated time: 30-60 seconds")

        if not self.core.confirm_operation("Execute quick test analysis"):
            return

        # Optimized parameters for quick testing
        params = {
            "seeds": 1000,
            "samples": 1000,
            "lmax": 8,
            "grid_size": 4,
            "description": "Quick connectivity test"
        }

        self.execute_analysis(params)

    def standard_analysis(self):
        """Execute standard analysis with parameter options"""
        print("\nStandard Analysis Configuration")
        print("=" * 40)
        
        # Analysis type selection
        print("\nAnalysis Type:")
        print("1. Correlation Analysis (VECTORIZED - FAST)")
        print("   • Processes ALL seeds per job in parallel")
        print("   • Pure correlation metrics (autocorrelation, spatial correlation)")
        print("   • 100x faster - recommended for large seed counts")
        print("   • Use for: Finding patterns across many seeds")
        print()
        print("2. Statistical Analysis (COMPREHENSIVE - DETAILED)")
        print("   • Processes 4 seeds per job for detailed testing")
        print("   • Chi-square, runs test, lag patterns, quality metrics")
        print("   • Slower but provides comprehensive quality assessment")
        print("   • Use for: PRNG quality testing and validation")
        print()
        
        analysis_mode = input("Select analysis type (1-2, default=1): ").strip()
        
        if analysis_mode == '2':
            analysis_type = 'statistical'
            print("\n✓ Statistical Analysis selected (comprehensive quality testing)")
        else:
            analysis_type = 'correlation'
            print("\n✓ Correlation Analysis selected (vectorized pattern finding)")
        
        print()
        print("Analysis target options:")
        print("1. General correlation analysis")
        print("2. Recent lottery pattern matching")
        print("3. Specific number targeting")
        print("4. Statistical randomness testing")

        target_choice = input("Select analysis target (1-4): ").strip()

        if target_choice == '1':
            params = {
                "seeds": 50000,
                "samples": 10000,
                "lmax": 32,
                "grid_size": 16,
                "description": "Standard correlation analysis",
                "analysis_type": analysis_type
            }
        elif target_choice == '2':
            params = {
                "seeds": 25000,
                "samples": 20000,
                "lmax": 16,
                "grid_size": 8,
                "description": "Recent pattern matching",
                "analysis_type": analysis_type
            }
        elif target_choice == '3':
            target_num = input("Enter target number (0-999): ").strip()
            try:
                target = int(target_num)
                if 0 <= target <= 999:
                    return self.execute_draw_match_analysis(target, 50000)
                else:
                    print("Number must be between 0-999")
                    input("Press Enter to continue...")
                    return
            except ValueError:
                print("Invalid number format")
                input("Press Enter to continue...")
                return
        elif target_choice == '4':
            params = {
                "seeds": 100000,
                "samples": 5000,
                "lmax": 64,
                "grid_size": 32,
                "description": "Randomness quality testing",
                "analysis_type": analysis_type
            }
        else:
            print("Invalid choice")
            input("Press Enter to continue...")
            return

        print(f"\nConfiguration: {params['description']}")
        print(f"Analysis Type: {analysis_type.upper()}")
        print(f"Seeds: {params['seeds']:,}, Samples: {params['samples']:,}")
        print(f"Correlation lag: {params['lmax']}, Grid size: {params['grid_size']}")
        
        if analysis_type == 'correlation':
            print("Estimated time: 2-5 minutes (vectorized)")
        else:
            print("Estimated time: 5-15 minutes (detailed per-seed analysis)")

        if self.core.confirm_operation("Execute standard analysis"):
            self.execute_analysis(params)

    def comprehensive_analysis(self):
        """Execute comprehensive analysis with advanced options"""
        print("\nComprehensive Analysis Configuration")
        print("=" * 40)

        print("Select comprehensive analysis type:")
        print("1. Deep correlation analysis (high lag, large grid)")
        print("2. Multi-target pattern search")
        print("3. Historical sequence reconstruction")
        print("4. Full spectrum PRNG testing")

        analysis_type = input("Select type (1-4): ").strip()

        if analysis_type == '1':
            params = {
                "seeds": 200000,
                "samples": 50000,
                "lmax": 128,
                "grid_size": 64,
                "description": "Deep correlation analysis"
            }
            estimated_time = "30-60 minutes"

        elif analysis_type == '2':
            recent_numbers = self.get_recent_lottery_numbers(5)
            if recent_numbers:
                print(f"Targeting recent numbers: {recent_numbers}")
                return self.execute_multi_target_search(recent_numbers, 150000)
            else:
                print("Could not load recent lottery numbers")
                input("Press Enter to continue...")
                return

        elif analysis_type == '3':
            params = {
                "seeds": 300000,
                "samples": 25000,
                "lmax": 256,
                "grid_size": 32,
                "description": "Historical sequence reconstruction"
            }
            estimated_time = "1-2 hours"

        elif analysis_type == '4':
            params = {
                "seeds": 500000,
                "samples": 20000,
                "lmax": 64,
                "grid_size": 48,
                "description": "Full spectrum PRNG testing"
            }
            estimated_time = "45-90 minutes"

        else:
            print("Invalid choice")
            input("Press Enter to continue...")
            return

        print(f"\nConfiguration: {params['description']}")
        print(f"Seeds: {params['seeds']:,}, Samples: {params['samples']:,}")
        print(f"Analysis depth: lmax={params['lmax']}, grid={params['grid_size']}")
        print(f"Estimated time: {estimated_time}")

        if self.core.confirm_operation("Execute comprehensive analysis"):
            self.execute_analysis(params)

    def production_analysis(self):
        """Execute production scale analysis with parameter optimization"""
        print("\nProduction Scale Analysis Configuration")
        print("=" * 40)
        print("WARNING: Very intensive operation")

        print("\nProduction analysis modes:")
        print("1. Maximum throughput test (speed optimization)")
        print("2. Deep pattern mining (accuracy optimization)")
        print("3. Exhaustive correlation mapping")
        print("4. Custom production parameters")

        mode = input("Select production mode (1-4): ").strip()

        if mode == '1':
            params = {
                "seeds": 2000000,
                "samples": 10000,
                "lmax": 16,
                "grid_size": 8,
                "description": "Maximum throughput test"
            }
            estimated_time = "15-30 minutes"

        elif mode == '2':
            params = {
                "seeds": 500000,
                "samples": 100000,
                "lmax": 64,
                "grid_size": 32,
                "description": "Deep pattern mining"
            }
            estimated_time = "2-4 hours"

        elif mode == '3':
            params = {
                "seeds": 1000000,
                "samples": 50000,
                "lmax": 128,
                "grid_size": 64,
                "description": "Exhaustive correlation mapping"
            }
            estimated_time = "3-6 hours"

        elif mode == '4':
            return self.custom_production_parameters()
        else:
            print("Invalid choice")
            input("Press Enter to continue...")
            return

        print(f"\nProduction Configuration: {params['description']}")
        print(f"Seeds: {params['seeds']:,}")
        print(f"Samples per seed: {params['samples']:,}")
        print(f"Total operations: {params['seeds'] * params['samples']:,}")
        print(f"Analysis parameters: lmax={params['lmax']}, grid={params['grid_size']}")
        print(f"Estimated time: {estimated_time}")
        print("\nThis will utilize all 26 GPUs across your cluster")

        if not self.core.confirm_operation("Execute production analysis", double_confirm=True):
            return

        self.execute_analysis(params)

    def custom_production_parameters(self):
        """Custom production parameter configuration"""
        print("\nCustom Production Parameters")
        print("-" * 30)

        try:
            seeds = int(input("Seeds to process (100K-10M): "))
            if not 100000 <= seeds <= 10000000:
                print("Seeds must be between 100,000 and 10,000,000")
                input("Press Enter to continue...")
                return

            samples = int(input("Samples per seed (1K-1M): "))
            if not 1000 <= samples <= 1000000:
                print("Samples must be between 1,000 and 1,000,000")
                input("Press Enter to continue...")
                return

            lmax = int(input("Correlation lag (1-512, default 64): ") or "64")
            grid_size = int(input("Grid size (4-128, default 32): ") or "32")

            total_ops = seeds * samples
            estimated_minutes = max(5, total_ops // 50000000)

            params = {
                "seeds": seeds,
                "samples": samples,
                "lmax": lmax,
                "grid_size": grid_size,
                "description": "Custom production analysis"
            }

            print(f"\nCustom Configuration:")
            print(f"  Seeds: {seeds:,}")
            print(f"  Samples: {samples:,}")
            print(f"  Total operations: {total_ops:,}")
            print(f"  Correlation lag: {lmax}")
            print(f"  Grid size: {grid_size}")
            print(f"  Estimated time: {estimated_minutes} minutes")

            if self.core.confirm_operation("Execute custom production analysis"):
                self.execute_analysis(params)

        except ValueError:
            print("Invalid input - must be numbers")
            input("Press Enter to continue...")

    def execute_draw_match_analysis(self, target_number: int, seeds: int):
        """Execute draw matching analysis for specific number"""
        print(f"\nDraw Matching Analysis: Target {target_number}")
        print(f"Testing {seeds:,} seeds for matches...")

        timestamp = int(time.time())
        output_file = f"results/draw_match_{target_number}_{timestamp}.json"
        os.makedirs("results", exist_ok=True)

        cmd = [
            "python3", "coordinator.py", "daily3.json",
            "-c", "distributed_config.json",
            "-s", str(seeds),
            "--draw-match", str(target_number),
            "-o", output_file
        ]

        self.execute_coordinator_command(cmd, f"draw match for {target_number}")

    def execute_multi_target_search(self, targets: list, seeds: int):
        """Execute search for multiple target numbers"""
        print(f"\nMulti-Target Search")
        print(f"Targets: {targets}")
        print(f"Seeds per target: {seeds // len(targets):,}")

        results = []
        for target in targets:
            print(f"\nSearching for {target}...")
            timestamp = int(time.time())
            output_file = f"results/multi_target_{target}_{timestamp}.json"

            cmd = [
                "python3", "coordinator.py", "daily3.json",
                "-c", "distributed_config.json",
                "-s", str(seeds // len(targets)),
                "--draw-match", str(target),
                "-o", output_file
            ]

            success = self.execute_coordinator_command(cmd, f"target {target}")
            results.append((target, success))

        print(f"\nMulti-target search complete:")
        for target, success in results:
            status = "SUCCESS" if success else "FAILED"
            print(f"  Target {target}: {status}")

        input("Press Enter to continue...")

    def get_recent_lottery_numbers(self, count: int) -> list:
        """Get recent lottery numbers from daily3.json"""
        try:
            with open(self.core.data_file, 'r') as f:
                data = json.load(f)

            recent = data[-count:] if len(data) >= count else data
            return [draw.get('draw', 0) for draw in recent if 'draw' in draw]
        except:
            return []

    def execute_coordinator_command(self, cmd: list, description: str) -> bool:
        """Execute coordinator command with real-time output display"""
        print(f"Executing: {' '.join(cmd)}")
        print("-" * 60)

        start_time = time.time()

        try:
            result = subprocess.run(cmd, timeout=7200)
            duration = time.time() - start_time

            print("-" * 60)

            if result.returncode == 0:
                print(f"SUCCESS! {description} completed in {duration:.1f} seconds")

                output_file = None
                for i, arg in enumerate(cmd):
                    if arg == '-o' and i + 1 < len(cmd):
                        output_file = cmd[i + 1]
                        break

                if output_file and os.path.exists(output_file):
                    file_size = os.path.getsize(output_file) / 1024
                    print(f"Results: {output_file} ({file_size:.1f} KB)")

                    try:
                        with open(output_file, 'r') as f:
                            results = json.load(f)

                        if 'metadata' in results:
                            meta = results['metadata']
                            print(f"Jobs: {meta.get('successful_jobs', 0)}/{meta.get('total_jobs', 0)}")
                            print(f"Runtime: {meta.get('total_runtime', 0):.1f}s")

                            if 'gpu_stats' in meta:
                                gpu_stats = meta['gpu_stats']
                                print(f"GPUs used: {gpu_stats.get('active_gpus', 'N/A')}")

                    except Exception:
                        pass

                return True
            else:
                print(f"ERROR: {description} failed (exit code {result.returncode})")
                return False

        except subprocess.TimeoutExpired:
            print(f"TIMEOUT: {description} exceeded 2-hour time limit")
            return False
        except KeyboardInterrupt:
            print(f"INTERRUPTED: {description} cancelled by user")
            return False
        except Exception as e:
            print(f"EXCEPTION: {e}")
            return False

    def execute_analysis(self, params: Dict[str, Any]):
        """Execute analysis with specified parameters"""
        timestamp = int(time.time())
        analysis_type = params['description'].lower().replace(' ', '_')
        output_file = f"results/{analysis_type}_{timestamp}.json"
        os.makedirs("results", exist_ok=True)

        cmd = [
            "python3", "coordinator.py", "daily3.json",
            "-c", "distributed_config.json",
            "-s", str(params["seeds"]),
            "-n", str(params["samples"]),
            "--lmax", str(params["lmax"]),
            "--grid-size", str(params["grid_size"]),
            "-o", output_file
        ]
        
        # Add analysis type parameter if specified
        if "analysis_type" in params:
            cmd.extend(["--analysis-type", params["analysis_type"]])

        print(f"\nExecuting {params['description']}...")
        print(f"Parameters: {params['seeds']:,} seeds, {params['samples']:,} samples")
        print(f"Analysis: lmax={params['lmax']}, grid={params['grid_size']}")
        if "analysis_type" in params:
            print(f"Type: {params['analysis_type'].upper()}")

        success = self.execute_coordinator_command(cmd, params['description'])

        if success:
            total_ops = params["seeds"] * params["samples"]
            self.core.log_operation("analysis", "SUCCESS", f"{params['description']} completed")

        input("Press Enter to continue...")

    def draw_matching_analysis(self):
        """Execute draw matching analysis with parameter selection"""
        print("\nDraw Matching Analysis")
        print("=" * 40)
        print("Find seeds that generate specific lottery numbers")

        try:
            target_number = int(input("Enter target lottery number (0-999): "))
            if not 0 <= target_number <= 999:
                print("Number must be between 0 and 999")
                input("Press Enter to continue...")
                return

            print("\nSearch intensity:")
            print("1. Quick search (10K seeds)")
            print("2. Standard search (100K seeds)")
            print("3. Deep search (500K seeds)")
            print("4. Exhaustive search (2M seeds)")

            intensity = input("Select intensity (1-4): ").strip()

            seed_counts = {'1': 10000, '2': 100000, '3': 500000, '4': 2000000}
            seeds = seed_counts.get(intensity, 100000)

            estimated_time = {10000: "30 seconds", 100000: "2-5 minutes",
                            500000: "10-20 minutes", 2000000: "30-60 minutes"}

            print(f"\nConfiguration:")
            print(f"Target: {target_number}")
            print(f"Seeds to test: {seeds:,}")
            print(f"Estimated time: {estimated_time.get(seeds, 'Unknown')}")

            if self.core.confirm_operation(f"Search for lottery number {target_number}"):
                self.execute_draw_match_analysis(target_number, seeds)

        except ValueError:
            print("Invalid input - must be a number")
            input("Press Enter to continue...")

    def connectivity_test(self):
        """Test cluster connectivity"""
        print("\nSystem Connectivity Test")
        print("=" * 35)

        if self.core.confirm_operation("Test cluster connectivity"):
            cmd = ["python3", "coordinator.py", "daily3.json", "-c", "distributed_config.json", "--test-only"]
            self.execute_coordinator_command(cmd, "connectivity test")

        input("Press Enter to continue...")

    def single_job_distribution(self):
        """Single job distribution"""
        print("Single job distribution functionality needs implementation")
        input("Press Enter to continue...")

    def batch_execution(self):
        """Batch execution"""
        print("Batch execution functionality needs implementation")
        input("Press Enter to continue...")

    def performance_benchmark(self):
        """Performance benchmark"""
        print("Running performance benchmark...")
        params = {
            "seeds": 10000,
            "samples": 5000,
            "lmax": 16,
            "grid_size": 8,
            "description": "Performance benchmark"
        }
        self.execute_analysis(params)

    def residue_sieve_analysis(self):
        """Execute residue sieve analysis for PRNG seed discovery"""
        print("\nResidue Sieve Analysis")
        print("=" * 40)
        
        # Dataset selection
        print("Datasets: 1=test_known.json  2=daily3.json  3=custom")
        dataset_choice = input("Select (1-3, default=1): ").strip()
        
        if dataset_choice == '2':
            dataset = 'daily3.json'
        elif dataset_choice == '3':
            dataset = input("Enter path: ").strip()
        else:
            dataset = 'test_known.json'
        
        # PRNG selection
        print("\nPRNGs: 1=xorshift32  2=pcg32  3=mt19937")
        prng_choice = input("Select (1-3, default=1): ").strip()
        
        if prng_choice == '2':
            prng_type = 'pcg32'
        elif prng_choice == '3':
            prng_type = 'mt19937'
        else:
            prng_type = 'xorshift32'
        
        # Seed range
        print("\nRange: 1=1K  2=100K  3=1M  4=10M  5=custom")
        range_choice = input("Select (1-5, default=1): ").strip()
        
        seeds_map = {'2': 100000, '3': 1000000, '4': 10000000, '5': 0}
        seeds = seeds_map.get(range_choice, 1000)
        
        if range_choice == '5':
            try:
                seeds = int(input("Enter seed count: ").strip())
            except:
                seeds = 1000
        
        # Summary
        print(f"\nDataset: {dataset}")
        print(f"PRNG: {prng_type}")
        print(f"Seeds: {seeds:,}")
        
        if not self.core.confirm_operation("Execute sieve"):
            return
        
        # Execute
        cmd = ['python3', 'coordinator.py', dataset, 
               '--method', 'residue_sieve', 
               '--prng-type', prng_type, 
               '--seeds', str(seeds)]
        
        print("\nRunning sieve...")
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("\n✓ Sieve completed")
        else:
            print("\n✗ Sieve failed")
        
        input("\nPress Enter...")
    def custom_analysis(self):
        """Custom analysis with full parameter control"""
        self.custom_production_parameters()

    def shutdown(self):
        """Module shutdown"""
        print(f"Shutting down {self.module_name}...")

    def create_analysis_filename(self, analysis_type, params):
        """Create readable filename for new analysis"""
        from results_manager import create_readable_filename

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        clean_type = analysis_type.lower().replace(' ', '-').replace('_', '-')

        param_str = ""
        if 'seeds' in params:
            seeds = params['seeds']
            if seeds >= 1000000:
                param_str += f"{seeds//1000000}M-seeds"
            elif seeds >= 1000:
                param_str += f"{seeds//1000}k-seeds"

        if 'samples' in params and params['samples'] > 0:
            samples = params['samples']
            if samples >= 1000000:
                param_str += f"-{samples//1000000}M-samples"
            elif samples >= 1000:
                param_str += f"-{samples//1000}k-samples"

        if 'target_number' in params:
            param_str += f"-target-{params['target_number']}"

        return f"results/{timestamp}_{clean_type}_{param_str}_PENDING.json"
