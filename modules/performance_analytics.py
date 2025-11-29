#!/usr/bin/env python3
"""
Performance Analytics Module - Analyzes system and job performance
Modular design for easy addition of new analysis types
"""

import sqlite3
import os
import json
import glob
from datetime import datetime

class PerformanceAnalytics:
    def __init__(self, core):
        self.core = core
        self.module_name = "Performance Analytics"
        self.db_file = 'prng_analysis.db'
        
        # Register analysis handlers - ADD NEW TYPES HERE
        self.analysis_handlers = {
            'database': self._analyze_database,
            'gap_aware': self._analyze_gap_aware,
            'best_window': self._analyze_best_window,
            'exhaustive_search': self._analyze_exhaustive_search,
            'results_directory': self._analyze_results_directory,
            'all_json': self._analyze_all_json_files,
        }

    def menu(self):
        """Display performance analytics menu"""
        while True:
            self.core.clear_screen()
            self.core.print_header()
            print(f"\n{self.module_name.upper()}")
            print("-" * 70)
            print("ANALYTICS OPTIONS:")
            print("  1. All Analytics Summary")
            print("  2. Database Statistics")
            print("  3. Gap-Aware Analysis Results")
            print("  4. Best Window Analysis")
            print("  5. Exhaustive Search Results")
            print("  6. Results Directory")
            print("  7. All JSON Files")
            print("  8. Back to Main Menu")
            print("-" * 70)
            
            choice = input("Select option (1-8): ").strip()
            
            if choice == '1':
                self._show_all_analytics()
            elif choice == '2':
                self._display_single('database')
            elif choice == '3':
                self._display_single('gap_aware')
            elif choice == '4':
                self._display_single('best_window')
            elif choice == '5':
                self._display_single('exhaustive_search')
            elif choice == '6':
                self._display_single('results_directory')
            elif choice == '7':
                self._display_single('all_json')
            elif choice == '8':
                break
            else:
                print("Invalid option")
                input("Press Enter to continue...")

    def _show_all_analytics(self):
        """Show all analytics in one view"""
        self.core.clear_screen()
        self.core.print_header()
        print(f"\n{self.module_name.upper()} - COMPREHENSIVE SUMMARY")
        print("=" * 70)
        
        for name, handler in self.analysis_handlers.items():
            print(f"\n {name.upper().replace('_', ' ')}")
            print("-" * 70)
            try:
                handler()
            except Exception as e:
                print(f"  Error: {e}")
        
        print("\n" + "=" * 70)
        input("Press Enter to continue...")

    def _display_single(self, analysis_type):
        """Display a single analysis type"""
        self.core.clear_screen()
        self.core.print_header()
        print(f"\n{analysis_type.upper().replace('_', ' ')}")
        print("=" * 70)
        
        try:
            self.analysis_handlers[analysis_type]()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 70)
        input("Press Enter to continue...")

    # =================================================================
    # ANALYSIS HANDLERS - ADD NEW METHODS BELOW AND REGISTER ABOVE
    # =================================================================

    def _analyze_database(self):
        """Database statistics"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [t[0] for t in cursor.fetchall()]
            
            total_rows = 0
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                if count > 0:
                    print(f"  {table}: {count} rows")
                    total_rows += count
            
            if total_rows == 0:
                print("  No data in database tables")
            
            db_size = os.path.getsize(self.db_file) / (1024 * 1024)
            print(f"\n  Database size: {db_size:.2f} MB")
            
            conn.close()
        except Exception as e:
            print(f"  Error: {e}")

    def _analyze_gap_aware(self):
        """Gap-aware analysis results"""
        gap_files = sorted(glob.glob('gap_aware_analysis_*.json'), reverse=True)
        
        if not gap_files:
            print("  No gap-aware analysis files found")
            return
        
        print(f"  Found {len(gap_files)} gap-aware analysis file(s)\n")
        
        for filepath in gap_files[:5]:  # Show last 5
            try:
                timestamp = os.path.getmtime(filepath)
                size = os.path.getsize(filepath) / 1024
                date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                print(f"  File: {os.path.basename(filepath)}")
                print(f"    Date: {date} | Size: {size:.1f} KB")
                
                if 'algorithm_results' in data:
                    for algo, result in data['algorithm_results'].items():
                        success = result.get('success', False)
                        status = "✓" if success else "✗"
                        print(f"    {status} {algo.upper()}: ", end="")
                        if success:
                            conf = result.get('confidence', 0)
                            print(f"confidence {conf:.2f}")
                        else:
                            print("no match")
                
                if 'windows_analyzed' in data:
                    print(f"    Windows analyzed: {data['windows_analyzed']}")
                
                print()
                
            except Exception as e:
                print(f"  Error reading {filepath}: {e}")

    def _analyze_best_window(self):
        """Best window data"""
        filepath = 'best_window_for_seed_search.json'
        
        if not os.path.exists(filepath):
            print("  No best window file found")
            return
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            print(f"  Window: {data.get('window', 'unknown')}")
            print(f"  Values: {data.get('count', 0)}")
            
            if 'statistics' in data:
                stats = data['statistics']
                print(f"  Mean: {stats.get('mean', 0):.2f}")
                print(f"  Min: {stats.get('min', 0)}")
                print(f"  Max: {stats.get('max', 0)}")
            
            print(f"  File size: {os.path.getsize(filepath) / 1024:.1f} KB")
            
        except Exception as e:
            print(f"  Error: {e}")

    def _analyze_exhaustive_search(self):
        """Exhaustive search results"""
        # Check for exhaustive search result files
        search_files = glob.glob('search_results_exhaustive_*.json')
        
        if not search_files:
            print("  No exhaustive search results found")
            return
        
        print(f"  Found {len(search_files)} exhaustive search result(s)\n")
        
        for filepath in sorted(search_files, reverse=True):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                print(f"  {os.path.basename(filepath)}")
                # Add specific exhaustive search result parsing here
                print(f"    {json.dumps(data, indent=4)[:200]}...")
                print()
            except Exception as e:
                print(f"  Error: {e}")

    def _analyze_results_directory(self):
        """Results directory contents"""
        results_dir = 'results'
        
        if not os.path.exists(results_dir):
            print("  No results directory found")
            return
        
        files = os.listdir(results_dir)
        
        if not files:
            print("  Results directory is empty")
            return
        
        print(f"  Found {len(files)} file(s) in results directory\n")
        
        for filename in sorted(files, reverse=True)[:10]:  # Show last 10
            filepath = os.path.join(results_dir, filename)
            size = os.path.getsize(filepath) / 1024
            timestamp = os.path.getmtime(filepath)
            date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"  {filename}")
            print(f"    Date: {date} | Size: {size:.1f} KB")

    def _analyze_all_json_files(self):
        """All JSON analysis files"""
        json_files = glob.glob('*.json')
        json_files = [f for f in json_files if not f.startswith('.')]
        
        if not json_files:
            print("  No JSON files found")
            return
        
        print(f"  Found {len(json_files)} JSON file(s)\n")
        
        # Group by type
        groups = {
            'gap_aware': [],
            'lottery': [],
            'config': [],
            'other': []
        }
        
        for f in json_files:
            if 'gap_aware' in f:
                groups['gap_aware'].append(f)
            elif 'daily' in f or 'lottery' in f:
                groups['lottery'].append(f)
            elif 'config' in f:
                groups['config'].append(f)
            else:
                groups['other'].append(f)
        
        for group_name, files in groups.items():
            if files:
                print(f"  {group_name.upper()}:")
                for f in sorted(files):
                    size = os.path.getsize(f) / 1024
                    print(f"    {f} ({size:.1f} KB)")
                print()

    def shutdown(self):
        """Cleanup"""
        pass
