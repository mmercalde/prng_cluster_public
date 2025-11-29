#!/usr/bin/env python3
"""
Result Viewer Module - View and analyze PRNG analysis results
"""

import os
import json
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
from collections import defaultdict


import numpy as np
from collections import defaultdict

class CorrelationPatternAnalyzer:
    """Correlation pattern analysis integrated into ResultViewer"""
    
    def __init__(self):
        self.composite_score_threshold = 0.05
        self.z_score_threshold = 2.0
    
    def analyze_correlation_patterns(self, results_data: dict) -> dict:
        """Analyze correlation patterns from results data"""
        results = results_data.get('results', [])
        if not results:
            return {'error': 'No results found in data'}
        
        # Analyze composite scores
        significant_composite = self._analyze_composite_scores(results)
        
        # Analyze lag correlations
        significant_lags = self._analyze_lag_correlations(results)
        
        # Analyze statistical properties
        property_anomalies = self._analyze_statistical_properties(results)
        
        # Cross-model validation
        cross_validation = self._cross_model_validation(results)
        
        return {
            'significant_composite_scores': significant_composite,
            'significant_lag_correlations': significant_lags,
            'statistical_property_anomalies': property_anomalies,
            'cross_model_validation': cross_validation,
            'summary': self._generate_summary(significant_composite, significant_lags, property_anomalies)
        }
    
    def _analyze_composite_scores(self, results: list) -> list:
        """Find seeds with unusually high composite scores"""
        significant_composite = []
        all_scores = [r.get('composite_score', 0) for r in results if 'composite_score' in r]
        
        if not all_scores:
            return significant_composite
            
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)
        
        for result in results:
            score = result.get('composite_score', 0)
            if score > self.composite_score_threshold:
                z_score = (score - mean_score) / std_score if std_score > 0 else 0
                significant_composite.append({
                    'job_id': result.get('job_id', 'unknown'),
                    'prng_type': result.get('prng_type', 'unknown'),
                    'mapping_type': result.get('mapping_type', 'unknown'),
                    'composite_score': score,
                    'z_score_significance': z_score,
                    'n_seeds': result.get('n_seeds', 0)
                })
        
        return sorted(significant_composite, key=lambda x: x['composite_score'], reverse=True)
    
    def _analyze_lag_correlations(self, results: list) -> list:
        """Analyze lag correlation patterns"""
        significant_lags = []
        
        for result in results:
            z_scores = result.get('z_scores', {})
            job_info = {
                'job_id': result.get('job_id', 'unknown'),
                'prng_type': result.get('prng_type', 'unknown'),
                'mapping_type': result.get('mapping_type', 'unknown')
            }
            
            for lag_type, z_value in z_scores.items():
                if abs(z_value) > self.z_score_threshold:
                    significant_lags.append({
                        **job_info,
                        'lag_type': lag_type,
                        'z_score': z_value,
                        'significance': 'High' if abs(z_value) > 3.0 else 'Moderate'
                    })
        
        return sorted(significant_lags, key=lambda x: abs(x['z_score']), reverse=True)
    
    def _analyze_statistical_properties(self, results: list) -> list:
        """Analyze detailed statistical properties for anomalies"""
        property_anomalies = []
        
        for result in results:
            properties = result.get('detailed_properties', [])
            if not properties:
                continue
                
            job_info = {
                'job_id': result.get('job_id', 'unknown'),
                'prng_type': result.get('prng_type', 'unknown'),
                'mapping_type': result.get('mapping_type', 'unknown')
            }
            
            for i, prop in enumerate(properties):
                mean_val = prop.get('mean', 0)
                std_val = prop.get('std', 0)
                min_val = prop.get('min', 0)
                max_val = prop.get('max', 0)
                
                if std_val > 0:
                    cv = std_val / mean_val if mean_val != 0 else float('inf')
                    range_ratio = (max_val - min_val) / std_val if std_val > 0 else 0
                    
                    if cv > 2.0 or range_ratio > 10.0:
                        property_anomalies.append({
                            **job_info,
                            'property_index': i,
                            'coefficient_variation': cv,
                            'range_ratio': range_ratio,
                            'anomaly_type': 'high_variance' if cv > 2.0 else 'wide_range'
                        })
        
        return property_anomalies
    
    def _cross_model_validation(self, results: list) -> dict:
        """Find patterns that appear across multiple PRNG models"""
        model_patterns = defaultdict(list)
        
        for result in results:
            prng_type = result.get('prng_type', 'unknown')
            composite_score = result.get('composite_score', 0)
            
            if composite_score > self.composite_score_threshold:
                model_patterns[prng_type].append({
                    'job_id': result.get('job_id'),
                    'composite_score': composite_score,
                    'mapping_type': result.get('mapping_type')
                })
        
        return dict(model_patterns)
    
    def _generate_summary(self, composite: list, lags: list, anomalies: list) -> dict:
        """Generate analysis summary"""
        return {
            'total_significant_composite': len(composite),
            'total_significant_lags': len(lags),
            'total_property_anomalies': len(anomalies),
            'has_significant_patterns': len(composite) > 0 or len(lags) > 0,
            'top_composite_score': max([c['composite_score'] for c in composite], default=0),
            'highest_z_score': max([abs(l['z_score']) for l in lags], default=0)
        }


class ResultViewer:
    """Result viewing and analysis functionality"""
    

    def _get_property_labels(self, result_data):
        """Get descriptive labels for statistical properties"""
        try:
            # Check if enhanced z-score data is available
            if 'z_scores_enhanced' in result_data:
                enhanced = result_data['z_scores_enhanced']
                if 'z_scores_detailed' in enhanced:
                    return list(enhanced['z_scores_detailed'].keys())
            
            # Check for enhanced lag analysis
            if 'enhanced_lag_analysis' in result_data:
                lag_data = result_data['enhanced_lag_analysis']
                top_corrs = lag_data.get('top_correlations', [])
                labels = ['Chi-square uniformity test']
                for i, (lag_num, abs_corr, corr_val) in enumerate(top_corrs[:3]):
                    labels.append(f"Lag-{lag_num} autocorrelation")
                while len(labels) < 4:
                    labels.append('Runs randomness test')
                return labels
            
            # Default descriptive labels
            return [
                'Chi-square uniformity test',
                'Lag-1 autocorrelation', 
                'Lag-5 autocorrelation',
                'Runs randomness test'
            ]
        except:
            return ['Property 0', 'Property 1', 'Property 2', 'Property 3']


    def _safe_numeric_compare(self, val1, val2, label="value"):
        """Safely compare two values that might be strings or numbers"""
        try:
            # Convert to float if they're strings
            num1 = float(val1) if isinstance(val1, str) else val1
            num2 = float(val2) if isinstance(val2, str) else val2
            
            difference = abs(num1 - num2)
            
            # Significance markers
            if difference > 2.0:
                marker = " ***"
            elif difference > 1.0:
                marker = " **"
            elif difference > 0.5:
                marker = " *"
            else:
                marker = ""
            
            return f"{num1:.3f} vs {num2:.3f} (diff: {difference:.3f}){marker}"
            
        except (ValueError, TypeError):
            # If conversion fails, just show as strings
            return f"{val1} vs {val2} (non-numeric)"

    def __init__(self, core):
        """Initialize with core system reference"""
        self.core = core
        self.module_name = "ResultViewer"
        self.viz_dir = "visualizations"
        os.makedirs(self.viz_dir, exist_ok=True)
    
    
    def correlation_analysis_menu(self):
        """Correlation pattern analysis menu"""
        analyzer = CorrelationPatternAnalyzer()
        
        while True:
            self.core.clear_screen()
            print("CORRELATION PATTERN ANALYSIS")
            print("=" * 40)
            print("1. Analyze Latest Results")
            print("2. Analyze Specific File")
            print("3. Set Analysis Thresholds")
            print("4. View Analysis History")
            print("5. Back to Results Menu")
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                self._analyze_latest_results(analyzer)
            elif choice == '2':
                self._analyze_specific_file(analyzer)
            elif choice == '3':
                self._set_analysis_thresholds(analyzer)
            elif choice == '4':
                self._view_analysis_history()
            elif choice == '5':
                break
            else:
                print("Invalid choice. Press Enter to continue...")
                input()
    
    def _analyze_latest_results(self, analyzer):
        """Analyze the most recent results file"""
        import glob
        
        # NEW FORMAT: Read from organized subdirectories
        results_files = []
        results_files.extend(glob.glob("results/summaries/*.txt"))
        results_files.extend(glob.glob("results/csv/*.csv"))
        results_files.extend(glob.glob("results/json/*.json"))
        if not results_files:
            print("No results files found in results/ directory")
            input("Press Enter to continue...")
            return
        
        # Exclude pattern analysis files
        data_files = [f for f in results_files if not f.endswith('_pattern_analysis.json')]
        if not data_files:
            print("No analysis result files found")
            input("Press Enter to continue...")
            return
        
        latest_file = max(data_files, key=os.path.getmtime)
        print(f"Analyzing latest file: {os.path.basename(latest_file)}")
        
        try:
            import json
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            analysis = analyzer.analyze_correlation_patterns(data)
            self._display_pattern_analysis(analysis, latest_file)
            
        except Exception as e:
            print(f"Error analyzing file: {e}")
        
        input("Press Enter to continue...")
    
    def _analyze_specific_file(self, analyzer):
        """Analyze a specific results file"""
        import glob
        
        results_files = glob.glob("results/*.json")
        # Exclude pattern analysis files
        data_files = [f for f in results_files if not f.endswith('_pattern_analysis.json')]
        
        if not data_files:
            print("No analysis result files found")
            input("Press Enter to continue...")
            return
        
        print("Available results files:")
        sorted_files = sorted(data_files, key=os.path.getmtime, reverse=True)[:10]
        for i, file in enumerate(sorted_files):
            filename = os.path.basename(file)
            size = os.path.getsize(file) / 1024
            mtime = datetime.fromtimestamp(os.path.getmtime(file))
            print(f"  {i+1}. {filename} ({size:.1f} KB) - {mtime.strftime('%m/%d %H:%M')}")
        
        try:
            choice = int(input("\nSelect file number: ")) - 1
            if 0 <= choice < len(sorted_files):
                selected_file = sorted_files[choice]
                
                import json
                with open(selected_file, 'r') as f:
                    data = json.load(f)
                
                analysis = analyzer.analyze_correlation_patterns(data)
                self._display_pattern_analysis(analysis, selected_file)
            else:
                print("Invalid selection")
        except (ValueError, IndexError, Exception) as e:
            print(f"Error: {e}")
        
        input("Press Enter to continue...")
    
    def _set_analysis_thresholds(self, analyzer):
        """Set analysis thresholds"""
        print("Current Analysis Thresholds:")
        print(f"  Composite Score Threshold: {analyzer.composite_score_threshold}")
        print(f"  Z-Score Threshold: {analyzer.z_score_threshold}")
        print("\nLower thresholds = more sensitive (more patterns detected)")
        print("Higher thresholds = more selective (only strong patterns)")
        
        try:
            new_composite = input(f"\nNew composite threshold [{analyzer.composite_score_threshold}]: ").strip()
            if new_composite:
                analyzer.composite_score_threshold = float(new_composite)
            
            new_zscore = input(f"New z-score threshold [{analyzer.z_score_threshold}]: ").strip()
            if new_zscore:
                analyzer.z_score_threshold = float(new_zscore)
            
            print("\nThresholds updated:")
            print(f"  Composite Score: {analyzer.composite_score_threshold}")
            print(f"  Z-Score: {analyzer.z_score_threshold}")
            
        except ValueError:
            print("Invalid input - thresholds unchanged")
        
        input("Press Enter to continue...")
    
    def _display_pattern_analysis(self, analysis, filename):
        """Display pattern analysis results"""
        if 'error' in analysis:
            print(f"Analysis Error: {analysis['error']}")
            return
        
        summary = analysis['summary']
        self.core.clear_screen()
        print("CORRELATION PATTERN ANALYSIS RESULTS")
        print("=" * 50)
        print(f"File: {os.path.basename(filename)}")
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 50)
        
        print("SUMMARY:")
        print(f"  Significant Composite Scores: {summary['total_significant_composite']}")
        print(f"  Significant Lag Correlations: {summary['total_significant_lags']}")
        print(f"  Statistical Property Anomalies: {summary['total_property_anomalies']}")
        
        if summary['has_significant_patterns']:
            print(f"  *** SIGNIFICANT PATTERNS DETECTED ***")
            print(f"  Top Composite Score: {summary['top_composite_score']:.6f}")
            print(f"  Highest Z-Score: {summary['highest_z_score']:.3f}")
        else:
            print(f"  No significant patterns detected with current thresholds")
        
        # Show top findings
        composite_scores = analysis['significant_composite_scores']
        if composite_scores:
            print(f"\nTOP SIGNIFICANT COMPOSITE SCORES:")
            for i, pattern in enumerate(composite_scores[:5]):
                print(f"  {i+1}. {pattern['prng_type']}+{pattern['mapping_type']}: "
                      f"{pattern['composite_score']:.6f} "
                      f"(Z: {pattern['z_score_significance']:.2f}, "
                      f"Seeds: {pattern['n_seeds']:,})")
        
        lag_correlations = analysis['significant_lag_correlations']
        if lag_correlations:
            print(f"\nTOP SIGNIFICANT LAG CORRELATIONS:")
            for i, lag in enumerate(lag_correlations[:5]):
                print(f"  {i+1}. {lag['prng_type']}+{lag['mapping_type']} "
                      f"{lag['lag_type']}: {lag['z_score']:.3f} ({lag['significance']})")
        
        # Cross-model validation
        cross_val = analysis['cross_model_validation']
        if cross_val:
            print(f"\nCROSS-MODEL VALIDATION:")
            for prng_type, patterns in cross_val.items():
                print(f"  {prng_type}: {len(patterns)} significant patterns")
        
        # Property anomalies
        anomalies = analysis['statistical_property_anomalies']
        if anomalies:
            print(f"\nSTATISTICAL PROPERTY ANOMALIES:")
            for i, anomaly in enumerate(anomalies[:3]):
                print(f"  {i+1}. {anomaly['prng_type']}+{anomaly['mapping_type']}: "
                      f"CV={anomaly['coefficient_variation']:.2f} ({anomaly['anomaly_type']})")
        
        # Save detailed results
        output_file = filename.replace('.json', '_pattern_analysis.json')
        try:
            import json
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"\nDetailed results saved to: {os.path.basename(output_file)}")
        except Exception as e:
            print(f"Error saving detailed results: {e}")
    
    def _view_analysis_history(self):
        """View previous pattern analyses"""
        import glob
        
        analysis_files = glob.glob("results/*_pattern_analysis.json")
        if not analysis_files:
            print("No previous pattern analyses found")
            print("Run some analyses first to build history")
            input("Press Enter to continue...")
            return
        
        self.core.clear_screen()
        print("PATTERN ANALYSIS HISTORY")
        print("=" * 40)
        
        sorted_files = sorted(analysis_files, key=os.path.getmtime, reverse=True)
        for i, file in enumerate(sorted_files[:15]):  # Show last 15
            filename = os.path.basename(file)
            size = os.path.getsize(file) / 1024
            mtime = datetime.fromtimestamp(os.path.getmtime(file))
            
            # Try to get summary info
            try:
                import json
                with open(file, 'r') as f:
                    data = json.load(f)
                summary = data.get('summary', {})
                patterns = summary.get('has_significant_patterns', False)
                status = "*** PATTERNS ***" if patterns else "no patterns"
            except:
                status = "unknown"
            
            print(f"  {i+1}. {filename}")
            print(f"      Size: {size:.1f} KB | Date: {mtime.strftime('%Y-%m-%d %H:%M')} | {status}")
        
        input("\nPress Enter to continue...")

    def menu(self):
        """Result viewer menu"""
        while True:
            self.core.clear_screen()
            self.core.print_header()
            print(f"\n{self.module_name.upper()}")
            print("-" * 35)
            print("RESULT VIEWING OPTIONS:")
            print("  1. Quick Result Viewer")
            print("  2. Browse All Results")
            print("  3. Search Results")
            print("  4. Compare Results")
            print("  5. Correlation Pattern Analysis")
            print("  6. Interactive Visualization")
            print("  7. Export Results")
            print("  8. Back to Main Menu")
            print("-" * 35)
            
            choice = input("Select option (1-8): ").strip()
            
            if choice == '1':
                self.quick_result_viewer()
            elif choice == '2':
                self.browse_all_results()
            elif choice == '3':
                self.search_results()
            elif choice == '4':
                self.compare_results()
            elif choice == '5':
                self.correlation_analysis_menu()
            elif choice == '6':
                self.interactive_visualization()
            elif choice == '7':
                self.export_results()
            elif choice == '8':
                break
            else:
                print("Invalid choice")
                input("Press Enter to continue...")
    
    def quick_result_viewer(self):
        """Quick terminal-based result viewer"""
        print("\nQuick Result Viewer")
        print("=" * 30)
        
        # Find most recent result file
        results_dir = "results"
        if not os.path.exists(results_dir):
            print("No results directory found")
            input("Press Enter to continue...")
            return
        
        # NEW FORMAT: Scan subdirectories
        result_files = []
        for subdir in ['summaries', 'csv', 'json', 'detailed', 'configs']:
            subdir_path = os.path.join(results_dir, subdir)
            if os.path.exists(subdir_path):
                files = os.listdir(subdir_path)
                result_files.extend([os.path.join(subdir, f) for f in files])
        if not result_files:
            print("No result files found")
            input("Press Enter to continue...")
            return
        
        # Sort by modification time
        result_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
        latest_file = result_files[0]
        file_path = os.path.join(results_dir, latest_file)
        
        print(f"Loading most recent result: {latest_file}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            self.display_results_summary(data, file_path)
            
            # Ask if user wants interactive visualization
            create_viz = input("\nOpen full interactive visualization? (y/N): ").strip().lower()
            if create_viz == 'y':
                self.create_interactive_visualization(data, latest_file)
                
        except Exception as e:
            print(f"Error loading results: {e}")
        
        input("Press Enter to continue...")
    
    def display_results_summary(self, data: Dict[str, Any], file_path: str):
        """Display results summary in terminal"""
        print(f"\n📊 ANALYSIS RESULTS FOR: {os.path.basename(file_path)}")
        print("=" * 60)
        
        # Detect result type and display accordingly
        if 'metadata' in data:
            # Coordinator results
            metadata = data['metadata']
            print("Type: Coordinator Analysis Results")
            print(f"Jobs: {metadata.get('successful_jobs', 0)}/{metadata.get('total_jobs', 0)} successful")
            print(f"Runtime: {metadata.get('total_runtime', 0):.2f} seconds")
            print(f"Nodes: {metadata.get('nodes_used', 0)} nodes, {metadata.get('total_gpus', 0)} GPUs")
            print(f"Hardware Optimized: {metadata.get('hardware_optimized', False)}")
            
            if 'results' in data and data['results']:
                print(f"Results: {len(data['results'])} detailed entries")
                
                # Show sample results if available
                sample_results = data['results'][:3]
                for i, result in enumerate(sample_results, 1):
                    if isinstance(result, dict):
                        score = result.get('score', result.get('composite_score', 'N/A'))
                        seed = result.get('seed', 'N/A')
                        print(f"  {i}. Seed {seed}: Score {score}")
            else:
                print("Results: No detailed results (metadata only)")
                
            if 'failed_jobs' in data and data['failed_jobs']:
                print(f"Failures: {len(data['failed_jobs'])} failed jobs")
        
        # File info
        file_size = os.path.getsize(file_path)
        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        print(f"File size: {file_size} bytes")
        print(f"Modified: {mod_time.strftime('%a %b %d %H:%M:%S %Y')}")
    
    def create_interactive_visualization(self, data: Dict[str, Any], filename: str):
        """Create interactive HTML visualization"""
        print("Creating visualization...")
        
        # Create visualization in visualizations directory
        viz_file = os.path.join(self.viz_dir, f"viz_{filename.replace('.json', '')}.html")
        
        html_content = self.generate_html_visualization(data, filename)
        
        try:
            with open(viz_file, 'w') as f:
                f.write(html_content)
            
            print(f"Opening browser visualization...")
            
            # Try multiple methods to open browser
            file_url = f"file://{os.path.abspath(viz_file)}"
            
            # Try xdg-open (Linux)
            try:
                subprocess.run(['xdg-open', viz_file], check=True, 
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("✅ Browser opened successfully")
                return
            except:
                pass
            
            # Try firefox directly
            try:
                subprocess.run(['firefox', viz_file], check=True,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("✅ Firefox opened successfully")
                return
            except:
                pass
            
            # Fallback - provide manual instructions
            print("❌ Could not open browser automatically")
            print(f"Manually open: {file_url}")
            print(f"Or run: firefox {viz_file}")
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
    
    def generate_html_visualization(self, data: Dict[str, Any], filename: str) -> str:
        """Generate HTML content for visualization"""
        data_json = json.dumps(data, indent=2, default=str)
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRNG Analysis Results - {filename}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        body {{
            font-family: 'Courier New', monospace;
            background: #0a0a0a;
            color: #00ff00;
            margin: 0;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{
            text-align: center;
            border: 2px solid #00ff00;
            padding: 20px;
            margin-bottom: 30px;
            background: rgba(0, 255, 0, 0.05);
        }}
        .header h1 {{
            color: #00ff00;
            font-size: 2.5em;
            margin: 0;
            text-shadow: 0 0 10px #00ff00;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            border: 1px solid #00ff00;
            padding: 20px;
            background: rgba(0, 255, 0, 0.02);
            border-radius: 5px;
        }}
        .stat-card h3 {{
            color: #00ff00;
            margin-top: 0;
            border-bottom: 1px solid #004400;
            padding-bottom: 10px;
        }}
        .chart-container {{
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #00ff00;
            background: rgba(0, 255, 0, 0.02);
        }}
        .chart-wrapper {{
            position: relative;
            height: 400px;
            margin: 20px 0;
        }}
        .json-viewer {{
            background: #111;
            border: 1px solid #333;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            max-height: 400px;
            overflow-y: auto;
        }}
        button {{
            background: #004400;
            color: #00ff00;
            border: 1px solid #00ff00;
            padding: 10px 20px;
            cursor: pointer;
            margin: 5px;
            font-family: 'Courier New', monospace;
        }}
        button:hover {{ background: #006600; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 PRNG ANALYSIS RESULTS</h1>
            <p>File: {filename} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="stats-grid" id="statsGrid">
            <!-- Stats populated by JavaScript -->
        </div>

        <div class="chart-container">
            <h2>📊 Performance Analysis</h2>
            <div class="chart-wrapper">
                <canvas id="performanceChart"></canvas>
            </div>
        </div>

        <div class="chart-container">
            <h2>📄 Raw Data</h2>
            <button onclick="toggleJsonViewer()">Toggle JSON Data</button>
            <div id="jsonViewer" class="json-viewer" style="display: none;">
                <pre id="jsonContent"></pre>
            </div>
        </div>
    </div>

    <script>
        const analysisData = {data_json};
        
        document.addEventListener('DOMContentLoaded', function() {{
            populateStats();
            createPerformanceChart();
            document.getElementById('jsonContent').textContent = JSON.stringify(analysisData, null, 2);
        }});
        
        function populateStats() {{
            const statsGrid = document.getElementById('statsGrid');
            
            if (analysisData.metadata) {{
                const meta = analysisData.metadata;
                const card = document.createElement('div');
                card.className = 'stat-card';
                card.innerHTML = `
                    <h3>⚡ Execution Summary</h3>
                    <p><strong>Jobs:</strong> ${{meta.successful_jobs || 0}}/${{meta.total_jobs || 0}} successful</p>
                    <p><strong>Runtime:</strong> ${{(meta.total_runtime || 0).toFixed(2)}} seconds</p>
                    <p><strong>Nodes:</strong> ${{meta.nodes_used || 0}} nodes</p>
                    <p><strong>GPUs:</strong> ${{meta.total_gpus || 0}} total</p>
                `;
                statsGrid.appendChild(card);
            }}
        }}
        
        function createPerformanceChart() {{
            const ctx = document.getElementById('performanceChart').getContext('2d');
            
            let labels = ['Successful', 'Failed', 'Runtime'];
            let data = [0, 0, 0];
            
            if (analysisData.metadata) {{
                const meta = analysisData.metadata;
                data = [
                    meta.successful_jobs || 0,
                    (meta.total_jobs || 0) - (meta.successful_jobs || 0),
                    (meta.total_runtime || 0).toFixed(2)
                ];
            }}
            
            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [{{
                        label: 'Performance Metrics',
                        data: data,
                        backgroundColor: ['rgba(0, 255, 0, 0.3)', 'rgba(255, 0, 0, 0.3)', 'rgba(0, 100, 255, 0.3)'],
                        borderColor: ['#00ff00', '#ff0000', '#0066ff'],
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ labels: {{ color: '#00ff00' }} }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            ticks: {{ color: '#00ff00' }},
                            grid: {{ color: '#004400' }}
                        }},
                        x: {{
                            ticks: {{ color: '#00ff00' }},
                            grid: {{ color: '#004400' }}
                        }}
                    }}
                }}
            }});
        }}
        
        function toggleJsonViewer() {{
            const viewer = document.getElementById('jsonViewer');
            viewer.style.display = viewer.style.display === 'none' ? 'block' : 'none';
        }}
    </script>
</body>
</html>"""
    
    def browse_all_results(self):
        """Browse all result files"""
        print("\nBrowse All Results")
        print("-" * 30)
        
        results_dir = "results"
        if not os.path.exists(results_dir):
            print("No results directory found")
            input("Press Enter to continue...")
            return
        
        result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        if not result_files:
            print("No result files found")
            input("Press Enter to continue...")
            return
        
        # Sort by modification time
        result_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
        
        print(f"Found {len(result_files)} result files:")
        
        for i, file in enumerate(result_files[:10], 1):  # Show first 10
            file_path = os.path.join(results_dir, file)
            file_size = os.path.getsize(file_path)
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            print(f"  {i}. {file} ({file_size} bytes, {mod_time.strftime('%m/%d %H:%M')})")
        
        if len(result_files) > 10:
            print(f"  ... and {len(result_files) - 10} more files")
        
        choice = input(f"\nSelect file to view (1-{min(10, len(result_files))}), or Enter to go back: ").strip()
        
        try:
            if choice:
                idx = int(choice) - 1
                if 0 <= idx < min(10, len(result_files)):
                    selected_file = result_files[idx]
                    file_path = os.path.join(results_dir, selected_file)
                    
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    self.display_results_summary(data, file_path)
                    
                    create_viz = input("\nCreate interactive visualization? (y/N): ").strip().lower()
                    if create_viz == 'y':
                        self.create_interactive_visualization(data, selected_file)
        except (ValueError, IndexError):
            print("Invalid selection")
        except Exception as e:
            print(f"Error loading file: {e}")
        
        input("Press Enter to continue...")
    
    def search_results(self):
        """Search results by criteria"""
        print("\nSearch Results - functionality coming soon")
        input("Press Enter to continue...")
    
    def compare_results(self):
        """Compare multiple result files"""
        # Get result files using same logic as browse_all_results
        results_dir = "results"
        if not os.path.exists(results_dir):
            print("\n❌ No results directory found")
            input("Press Enter to continue...")
            return
            
        result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        if len(result_files) < 2:
            print("\n❌ Need at least 2 result files to compare")
            input("Press Enter to continue...")
            return
        
        # Sort by modification time
        result_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
        
        print("\n" + "="*50)
        print("COMPARE RESULTS")
        print("="*50)
        print("Available result files:")
        
        # Group files by type
        raw_files = [f for f in result_files if not f.endswith('_pattern_analysis.json')]
        pattern_files = [f for f in result_files if f.endswith('_pattern_analysis.json')]
        
        print("\nRAW ANALYSIS FILES (correlation data):")
        print("-" * 40)
        file_index = 1
        raw_indices = {}
        for filename in raw_files:
            filepath = os.path.join(results_dir, filename)
            size = os.path.getsize(filepath) / 1024  # KB
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            print(f"  {file_index}. {filename}")
            print(f"      Size: {size:.1f} KB | Date: {mtime.strftime('%Y-%m-%d %H:%M')}")
            raw_indices[file_index] = filename
            file_index += 1
        
        if not raw_files:
            print("  (No raw analysis files found)")
        
        print("\nPATTERN ANALYSIS FILES (processed summaries):")
        print("-" * 45)
        pattern_indices = {}
        for filename in pattern_files:
            filepath = os.path.join(results_dir, filename)
            size = os.path.getsize(filepath) / 1024  # KB
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            print(f"  {file_index}. {filename}")
            print(f"      Size: {size:.1f} KB | Date: {mtime.strftime('%Y-%m-%d %H:%M')}")
            pattern_indices[file_index] = filename
            file_index += 1
        
        if not pattern_files:
            print("  (No pattern analysis files found)")
        
        print("\n" + "="*50)
        print("NOTE: Only compare files of the same type for meaningful results")
        print("Raw files contain correlation measurements")
        print("Pattern files contain processed analysis summaries")
        print("="*50)
        
        # Create combined index mapping
        all_indices = {**raw_indices, **pattern_indices}
        max_index = max(all_indices.keys()) if all_indices else 0
        
        # Get file selections
        try:
            selection1 = input("\nSelect first file (1-{}): ".format(max_index)).strip()
            selection2 = input("Select second file (1-{}): ".format(max_index)).strip()
            
            sel1 = int(selection1)
            sel2 = int(selection2)
            
            if sel1 not in all_indices or sel2 not in all_indices:
                raise ValueError("Invalid selection")
                
            if sel1 == sel2:
                print("\n❌ Cannot compare a file with itself")
                input("Press Enter to continue...")
                return
            
            file1_name = all_indices[sel1]
            file2_name = all_indices[sel2]
            
            # Check compatibility
            file1_is_pattern = file1_name.endswith('_pattern_analysis.json')
            file2_is_pattern = file2_name.endswith('_pattern_analysis.json')
            
            if file1_is_pattern != file2_is_pattern:
                print("\n⚠️  WARNING: You're comparing different file types!")
                print(f"  File 1: {'Pattern analysis' if file1_is_pattern else 'Raw analysis'}")
                print(f"  File 2: {'Pattern analysis' if file2_is_pattern else 'Raw analysis'}")
                print("\nThis comparison may not provide meaningful results.")
                proceed = input("Continue anyway? (y/N): ").strip().lower()
                if proceed != 'y':
                    return
                
        except (ValueError, IndexError):
            print("\n❌ Invalid selection")
            input("Press Enter to continue...")
            return
        
        # Load and compare the files
        file1_path = os.path.join(results_dir, file1_name)
        file2_path = os.path.join(results_dir, file2_name)
        
        try:
            with open(file1_path, 'r') as f:
                data1 = json.load(f)
            with open(file2_path, 'r') as f:
                data2 = json.load(f)
        except Exception as e:
            print(f"\n❌ Error loading files: {e}")
            input("Press Enter to continue...")
            return
        
        # Perform detailed comparison
        self._display_detailed_comparison(file1_name, data1, file2_name, data2)


    def _display_detailed_comparison(self, name1, data1, name2, data2):
        """Display comprehensive comparison with detailed statistics"""
        print("\n" + "="*80)
        print("DETAILED RESULT COMPARISON")
        print("="*80)
        print(f"File 1: {name1}")
        print(f"File 2: {name2}")
        print("-" * 80)
        
        # Determine file types
        is_pattern1 = name1.endswith('_pattern_analysis.json')
        is_pattern2 = name2.endswith('_pattern_analysis.json')
        
        if is_pattern1 and is_pattern2:
            self._compare_pattern_files(data1, data2)
        elif not is_pattern1 and not is_pattern2:
            self._compare_raw_files(data1, data2)
        else:
            self._compare_mixed_files(data1, data2, is_pattern1)
        
        input("\nPress Enter to continue...")
    
    def _compare_raw_files(self, data1, data2):
        """Compare two raw analysis files with correlation data"""
        print("\nRAW ANALYSIS COMPARISON")
        print("=" * 40)
        
        # Extract metadata
        meta1 = data1.get('metadata', {})
        meta2 = data2.get('metadata', {})
        results1 = data1.get('results', [])
        results2 = data2.get('results', [])
        
        print("\nMETADATA COMPARISON:")
        print("-" * 25)
        key_params = ['total_jobs', 'successful_jobs', 'failed_jobs', 'total_runtime', 'nodes_used', 'total_gpus']
        for param in key_params:
            val1 = meta1.get(param, 'N/A')
            val2 = meta2.get(param, 'N/A')
            match = "✓" if val1 == val2 else "✗"
            print(f"  {param.ljust(15)}: {str(val1).ljust(8)} vs {str(val2).ljust(8)} [{match}]")
        
        print("\nRESULT ANALYSIS:")
        print("-" * 20)
        print(f"  Result count        : {len(results1)} vs {len(results2)}")
        
        if results1 and results2:
            result1 = results1[0]  # Assuming single result per file
            result2 = results2[0]
            
            # Basic comparison
            print(f"  PRNG Type           : {result1.get('prng_type', 'N/A')} vs {result2.get('prng_type', 'N/A')}")
            print(f"  Mapping Type        : {result1.get('mapping_type', 'N/A')} vs {result2.get('mapping_type', 'N/A')}")
            print(f"  Seeds Processed     : {result1.get('n_seeds', 'N/A')} vs {result2.get('n_seeds', 'N/A')}")
            print(f"  Composite Score     : {result1.get('composite_score', 0):.6f} vs {result2.get('composite_score', 0):.6f}")
            
            # Detailed properties comparison
            props1 = result1.get('detailed_properties', {})
            props2 = result2.get('detailed_properties', {})
            
            if props1 and props2:
                print("\nDETAILED PROPERTIES COMPARISON:")
                print("-" * 35)
                
                # Check if detailed_properties is a dict or list
                if isinstance(props1, dict) and isinstance(props2, dict):
                    # Handle dictionary format
                    common_props = set(props1.keys()) & set(props2.keys())
                    if common_props:
                        print("  Statistical Properties:")
                        for prop in sorted(common_props):
                            val1 = props1[prop]
                            val2 = props2[prop]
                            
                            if isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
                                if len(val1) == len(val2):
                                    try:
                                        # Convert to float and calculate statistics
                                        nums1 = [float(v) for v in val1 if v is not None]
                                        nums2 = [float(v) for v in val2 if v is not None]
                                        
                                        if nums1 and nums2:
                                            mean1 = sum(nums1) / len(nums1)
                                            mean2 = sum(nums2) / len(nums2)
                                            max1 = max(nums1)
                                            max2 = max(nums2)
                                            min1 = min(nums1)
                                            min2 = min(nums2)
                                            
                                            print(f"    {prop.ljust(20)}:")
                                            print(f"      Mean          : {mean1:8.4f} vs {mean2:8.4f}")
                                            print(f"      Max           : {max1:8.4f} vs {max2:8.4f}")
                                            print(f"      Min           : {min1:8.4f} vs {min2:8.4f}")
                                            print(f"      Range         : {max1-min1:8.4f} vs {max2-min2:8.4f}")
                                        else:
                                            print(f"    {prop.ljust(20)}: Unable to calculate (non-numeric data)")
                                    except (ValueError, TypeError):
                                        print(f"    {prop.ljust(20)}: Unable to compare (mixed data types)")
                            else:
                                # Simple value comparison
                                try:
                                    fval1 = float(val1) if val1 is not None else 0.0
                                    fval2 = float(val2) if val2 is not None else 0.0
                                    print(f"    {prop.ljust(20)}: {fval1:8.4f} vs {fval2:8.4f}")
                                except:
                                    print(f"    {prop.ljust(20)}: {str(val1).ljust(8)} vs {str(val2).ljust(8)}")
                else:
                    print("  Properties data format not supported for detailed comparison")
                    print(f"  Props1 type: {type(props1).__name__}")
                    print(f"  Props2 type: {type(props2).__name__}")
            
            # Z-scores comparison (moved outside detailed_properties check)
            z1 = result1.get('z_scores', [])
            z2 = result2.get('z_scores', [])
            if z1 and z2:
                print("\n  Z-Score Analysis:")
                for i, (zs1, zs2) in enumerate(zip(z1, z2)):
                    try:
                        # Convert to float in case they're stored as strings
                        val1 = float(zs1) if zs1 is not None else 0.0
                        val2 = float(zs2) if zs2 is not None else 0.0
                        diff = abs(val1 - val2)
                        significance = "***" if diff > 2.0 else "**" if diff > 1.0 else "*" if diff > 0.5 else ""
                        print(f"    Property {i:2d}    : {val1:8.3f} vs {val2:8.3f} (Δ={diff:6.3f}) {significance}")
                    except (ValueError, TypeError):
                        # Handle non-numeric values
                        print(f"    Property {i:2d}    : {str(zs1).ljust(8)} vs {str(zs2).ljust(8)} (non-numeric)")
        
        # Performance comparison
        perf1 = data1.get('performance_summary', {})
        perf2 = data2.get('performance_summary', {})
        if perf1 and perf2:
            print("\nPERFORMANCE COMPARISON:")
            print("-" * 25)
            print(f"  Total Seeds Processed : {perf1.get('total_seeds_processed', 0)} vs {perf2.get('total_seeds_processed', 0)}")
            print(f"  Average Job Runtime   : {perf1.get('average_job_runtime', 0):.3f}s vs {perf2.get('average_job_runtime', 0):.3f}s")
            print(f"  Fastest GPU Runtime   : {perf1.get('fastest_gpu_runtime', 0):.3f}s vs {perf2.get('slowest_gpu_runtime', 0):.3f}s")
    
    def _compare_pattern_files(self, data1, data2):
        """Compare two pattern analysis files"""
        print("\nPATTERN ANALYSIS COMPARISON")
        print("=" * 35)
        
        # Summary comparison
        sum1 = data1.get('summary', {})
        sum2 = data2.get('summary', {})
        
        print("\nSUMMARY COMPARISON:")
        print("-" * 20)
        summary_keys = ['total_significant_composite', 'total_significant_lags', 'total_property_anomalies', 
                       'has_significant_patterns', 'top_composite_score', 'highest_z_score']
        
        for key in summary_keys:
            val1 = sum1.get(key, 'N/A')
            val2 = sum2.get(key, 'N/A')
            if isinstance(val1, float) and isinstance(val2, float):
                print(f"  {key.ljust(25)}: {val1:8.4f} vs {val2:8.4f}")
            else:
                print(f"  {key.ljust(25)}: {str(val1).ljust(8)} vs {str(val2).ljust(8)}")
        
        # Significant composite scores
        comp1 = data1.get('significant_composite_scores', [])
        comp2 = data2.get('significant_composite_scores', [])
        
        print("\nSIGNIFICANT COMPOSITE SCORES:")
        print("-" * 30)
        print(f"  Count                 : {len(comp1)} vs {len(comp2)}")
        
        if comp1 or comp2:
            max_items = max(len(comp1), len(comp2))
            for i in range(max_items):
                item1 = comp1[i] if i < len(comp1) else {}
                item2 = comp2[i] if i < len(comp2) else {}
                
                score1 = item1.get('composite_score', 0)
                score2 = item2.get('composite_score', 0)
                z1 = item1.get('z_score_significance', 0)
                z2 = item2.get('z_score_significance', 0)
                
                print(f"  Item {i+1}:")
                print(f"    Composite Score   : {score1:8.6f} vs {score2:8.6f}")
                print(f"    Z-Score           : {z1:8.3f} vs {z2:8.3f}")
        
        # Lag correlations
        lag1 = data1.get('significant_lag_correlations', [])
        lag2 = data2.get('significant_lag_correlations', [])
        
        print("\nSIGNIFICANT LAG CORRELATIONS:")
        print("-" * 30)
        print(f"  Count                 : {len(lag1)} vs {len(lag2)}")
        
        if lag1 or lag2:
            max_lags = max(len(lag1), len(lag2))
            for i in range(min(5, max_lags)):  # Show first 5
                l1 = lag1[i] if i < len(lag1) else {}
                l2 = lag2[i] if i < len(lag2) else {}
                
                print(f"  Lag {i+1}:")
                print(f"    Correlation       : {l1.get('correlation', 0):8.6f} vs {l2.get('correlation', 0):8.6f}")
                print(f"    Lag Distance      : {l1.get('lag', 'N/A')} vs {l2.get('lag', 'N/A')}")
        
        # Statistical anomalies
        anom1 = data1.get('statistical_property_anomalies', [])
        anom2 = data2.get('statistical_property_anomalies', [])
        
        print("\nSTATISTICAL ANOMALIES:")
        print("-" * 20)
        print(f"  Count                 : {len(anom1)} vs {len(anom2)}")
        
        if anom1 or anom2:
            # Group by anomaly type
            types1 = {}
            types2 = {}
            
            for anom in anom1:
                atype = anom.get('anomaly_type', 'unknown')
                types1[atype] = types1.get(atype, 0) + 1
            
            for anom in anom2:
                atype = anom.get('anomaly_type', 'unknown')
                types2[atype] = types2.get(atype, 0) + 1
            
            all_types = set(types1.keys()) | set(types2.keys())
            for atype in sorted(all_types):
                count1 = types1.get(atype, 0)
                count2 = types2.get(atype, 0)
                print(f"  {atype.ljust(15)}: {count1} vs {count2}")
    
    def _compare_mixed_files(self, data1, data2, first_is_pattern):
        """Handle comparison between different file types"""
        print("\nMIXED FILE TYPE COMPARISON")
        print("=" * 30)
        print("⚠️  Warning: Comparing different file types")
        print("This comparison has limited meaningfulness.\n")
        
        if first_is_pattern:
            print("File 1: Pattern Analysis (processed summary)")
            print("File 2: Raw Analysis (correlation data)")
            pattern_data = data1
            raw_data = data2
        else:
            print("File 1: Raw Analysis (correlation data)")
            print("File 2: Pattern Analysis (processed summary)")
            pattern_data = data2
            raw_data = data1
        
        # Try to extract comparable information
        pattern_summary = pattern_data.get('summary', {})
        raw_results = raw_data.get('results', [])
        
        print("\nLIMITED COMPARISON:")
        print("-" * 20)
        
        if raw_results:
            raw_composite = raw_results[0].get('composite_score', 0)
            pattern_composite = pattern_summary.get('top_composite_score', 0)
            print(f"Top Composite Score : {raw_composite:.6f} (raw) vs {pattern_composite:.6f} (pattern)")
        
        print("\nFor meaningful comparison, compare files of the same type.")
    def interactive_visualization(self):
        """Create interactive visualizations"""
        print("\nInteractive Visualization")
        print("-" * 30)
        
        # This is the same as quick_result_viewer but focuses on visualization
        self.quick_result_viewer()
    
    def export_results(self):
        """Export results in various formats"""
        print("\nExport Results - functionality coming soon")
        input("Press Enter to continue...")
    
    def shutdown(self):
        """Module shutdown"""
        print(f"Shutting down {self.module_name}...")


# === ADDED: renderer for MT state reconstruction ===
def _render_state_reconstruction(self, result: dict):
    ver = result.get("verification") or {}
    print("== MT State Reconstruction ==")
    print("Engine:", result.get("engine"))
    print("Inputs:", result.get("inputs"))
    print("Verified:", ver.get("verified"), " Match rate:", ver.get("match_rate"))
    state_head = result.get("reconstructed_state") or result.get("reconstructed_state_head") or []
    if state_head:
        print("State preview (first 10 words):", state_head[:10])
