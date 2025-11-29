#!/usr/bin/env python3
"""
Visualization Manager Module - Complete Fixed Implementation
Creates interactive visualizations with real GPU data from 26-GPU cluster
"""

import os
import json
import time
import webbrowser
import subprocess
import tempfile
from datetime import datetime
from typing import Dict, List, Any

# Import our GPU data collector
import sys
sys.path.append('.')
from gpu_data_collector import GPUDataCollector

class VisualizationManager:
    def __init__(self, core):
        self.core = core
        self.module_name = "VisualizationManager"
        self.output_dir = "visualizations"
        self.gpu_collector = GPUDataCollector()
        os.makedirs(self.output_dir, exist_ok=True)
    def analyze_correlation_data(self, data):
        """Extract correlation data for visualization"""
        analysis = {
            'correlations': [],
            'z_scores': [],
            'stats': {},
            'metadata': {}
        }
        
        try:
            results = data.get('results', [])
            if results:
                result = results[0]
                
                # Extract basic data
                analysis['metadata'] = {
                    'composite_score': result.get('composite_score', 0),
                    'seeds': result.get('n_seeds', 0),
                    'runtime': result.get('runtime', 0),
                    'prng_type': result.get('prng_type', 'unknown')
                }
                
                # Z-scores
                if 'z_scores' in result:
                    z_scores = result['z_scores']
                    if isinstance(z_scores, list):
                        analysis['z_scores'] = z_scores
                
                # Enhanced lag analysis if available
                if 'enhanced_lag_analysis' in result:
                    lag_data = result['enhanced_lag_analysis']
                    correlations = lag_data.get('top_correlations', [])
                    for lag_num, abs_corr, corr_val in correlations[:10]:
                        analysis['correlations'].append({
                            'lag': lag_num,
                            'correlation': corr_val,
                            'magnitude': abs_corr
                        })
                
                # Basic lag correlations fallback
                elif 'detailed_properties' in result and result['detailed_properties']:
                    props = result['detailed_properties'][0]
                    lag_corrs = props.get('lag_correlations', {})
                    for lag_name, corr_val in lag_corrs.items():
                        if lag_name.startswith('lag_'):
                            lag_num = int(lag_name.split('_')[1])
                            analysis['correlations'].append({
                                'lag': lag_num,
                                'correlation': corr_val,
                                'magnitude': abs(corr_val)
                            })
                    
                    analysis['stats'] = {
                        'chi2_p': props.get('chi2_p_value', 0),
                        'length': props.get('length', 0)
                    }
        
        except Exception as e:
            print(f"Error analyzing data: {e}")
        
        return analysis

    def create_correlation_chart_html(self, analysis, filename):
        """Create correlation visualization HTML"""
        timestamp = int(time.time())
        html_file = f"correlation_chart_{timestamp}.html"
        filepath = os.path.join(self.output_dir, html_file)
        
        correlations = analysis.get('correlations', [])
        z_scores = analysis.get('z_scores', [])
        metadata = analysis.get('metadata', {})
        
        # Prepare chart data
        lag_numbers = [c['lag'] for c in correlations]
        correlation_values = [c['correlation'] for c in correlations]
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Correlation Analysis: {filename}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ background: #1a1a1a; color: white; font-family: Arial; padding: 20px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }}
        .metric {{ background: #333; padding: 15px; border-radius: 10px; text-align: center; }}
        .chart-container {{ background: #333; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        canvas {{ max-height: 400px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Correlation Pattern Analysis</h1>
        <h2>{filename}</h2>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>{metadata.get('seeds', 0)}</h3>
            <p>Seeds Analyzed</p>
        </div>
        <div class="metric">
            <h3>{len(correlations)}</h3>
            <p>Lag Distances</p>
        </div>
        <div class="metric">
            <h3>{metadata.get('composite_score', 0):.4f}</h3>
            <p>Composite Score</p>
        </div>
        <div class="metric">
            <h3>{metadata.get('runtime', 0):.1f}s</h3>
            <p>Runtime</p>
        </div>
    </div>
    
    <div class="chart-container">
        <h3>Lag Correlation Analysis</h3>
        <canvas id="correlationChart"></canvas>
    </div>
    
    <div class="chart-container">
        <h3>Z-Score Analysis</h3>
        <canvas id="zScoreChart"></canvas>
    </div>
    
    <script>
        Chart.defaults.color = 'white';
        
        // Correlation Chart
        new Chart(document.getElementById('correlationChart'), {{
            type: 'line',
            data: {{
                labels: {lag_numbers},
                datasets: [{{
                    label: 'Correlation',
                    data: {correlation_values},
                    borderColor: '#4fc3f7',
                    backgroundColor: 'rgba(79, 195, 247, 0.1)',
                    fill: true
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{ title: {{ display: true, text: 'Lag Distance' }} }},
                    y: {{ title: {{ display: true, text: 'Correlation' }} }}
                }}
            }}
        }});
        
        // Z-Score Chart
        new Chart(document.getElementById('zScoreChart'), {{
            type: 'bar',
            data: {{
                labels: ['Chi2', 'Lag-1', 'Lag-5', 'Runs'],
                datasets: [{{
                    label: 'Z-Score',
                    data: {z_scores},
                    backgroundColor: ['#4caf50', '#ff9800', '#f44336', '#2196f3']
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{ title: {{ display: true, text: 'Z-Score Value' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        return filepath



    def menu(self):
        """Main visualization menu"""
        while True:
            self.core.clear_screen()
            self.core.print_header()
            print(f"\n{self.module_name.upper()}")
            print("-" * 50)
            print("RESULT VISUALIZATION:")
            print("  1. Quick Result Viewer")
            print("  2. Load & Analyze Result File")
            print("  3. Compare Multiple Results")
            print("  4. Generate Report")
            print("\nREAL-TIME MONITORING:")
            print("  5. Live Cluster Monitor")
            print("  6. GPU Performance Dashboard")
            print("  7. Network Status Monitor")
            print("\nTOOLS:")
            print("  8. Create Custom Visualization")
            print("  9. Export Graphs")
            print(" 10. System Requirements Check")
            print(" 11. Back to Main Menu")
            print("-" * 50)

            choice = input("Select option (1-11): ").strip()

            if choice == '1':
                self.quick_result_viewer()
            elif choice == '2':
                self.load_analyze_result()
            elif choice == '3':
                self.compare_multiple_results()
            elif choice == '4':
                self.generate_report()
            elif choice == '5':
                self.live_cluster_monitor()
            elif choice == '6':
                self.gpu_performance_dashboard()
            elif choice == '7':
                self.network_status_monitor()
            elif choice == '8':
                self.create_custom_visualization()
            elif choice == '9':
                self.export_graphs()
            elif choice == '10':
                self.system_requirements_check()
            elif choice == '11':
                break
            else:
                print("Invalid choice")
                input("Press Enter to continue...")

    def quick_result_viewer(self):
        """Quick visualization of latest result"""
        print("\nQuick Result Viewer")
        print("=" * 30)

        results_dir = "results"
        if not os.path.exists(results_dir):
            print("No results directory found")
            input("Press Enter to continue...")
            return

        json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        if not json_files:
            print("No result files found")
            input("Press Enter to continue...")
            return

        # Get latest file
        latest_file = max(json_files, key=lambda x: os.path.getmtime(os.path.join(results_dir, x)))
        filepath = os.path.join(results_dir, latest_file)

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            print(f"Latest result: {latest_file}")
            self.display_result_summary(data)

            if input("Create HTML visualization? (y/N): ").lower() == 'y':
                # Detect analysis type and create appropriate visualization
                analysis_type = self._detect_analysis_type(data)
                if analysis_type == 'draw_matching':
                    viz_file = f"visualizations/quick_draw_match_viz_{int(time.time())}.html"
                    self.create_simple_draw_match_viz(data, viz_file)
                    print(f"Chart.js visualization created: {viz_file}")
                else:
                    html_file = self.create_result_html(data, latest_file)
                    print(f"Basic visualization created: {html_file}")
                print("Available via web server")

        except Exception as e:
            print(f"Error loading result: {e}")

        input("Press Enter to continue...")

    def load_analyze_result(self):
        """Load and analyze specific result file"""
        print("\nLoad & Analyze Result File")
        print("=" * 30)
        
        results_dir = "results"
        if not os.path.exists(results_dir):
            print("No results directory found")
            input("Press Enter to continue...")
            return
            
        json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        if not json_files:
            print("No result files found")
            input("Press Enter to continue...")
            return
        
        print("Available result files:")
        for i, filename in enumerate(json_files[-10:], 1):  # Show last 10 files
            filepath = os.path.join(results_dir, filename)
            size = os.path.getsize(filepath)
            print(f"   {i}. {filename}")
            print(f"      Size: {size/1024:.1f} KB")
        
        try:
            choice = int(input("Select file number: ")) - 1
            if 0 <= choice < len(json_files[-10:]):
                filename = json_files[-10:][choice]
                filepath = os.path.join(results_dir, filename)
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                print(f"\nAnalyzing: {filename}")
                self.display_result_summary(data)
                
                if input("Create visualization? (y/N): ").lower() == 'y':
                    # Use the same detection and routing logic as option 1
                    analysis_type = self._detect_analysis_type(data)
                    if analysis_type == 'draw_matching':
                        viz_file = f"visualizations/load_draw_match_viz_{int(time.time())}.html"
                        self.create_simple_draw_match_viz(data, viz_file)
                        print(f"Chart.js visualization created: {viz_file}")
                    else:
                        html_file = self.create_result_html(data, filename)
                        print(f"Basic visualization created: {html_file}")
                    print("Available via web server")
            else:
                print("Invalid selection")
        except (ValueError, IndexError):
            print("Invalid input")
        except Exception as e:
            print(f"Error: {e}")
        
        input("Press Enter to continue...")
    def _detect_analysis_type(self, data):
        """Detect analysis type from data"""
        try:
            results = data.get('results', [])
            for result in results:
                if isinstance(result, dict):
                    if 'advanced_draw_matching' in str(result.get('method', '')):
                        return 'draw_matching'
                    if 'match_rate' in result:
                        return 'draw_matching'
            return 'correlation'
        except:
            return 'correlation'

    
    def create_simple_draw_match_viz(self, data, output_file):
        """Create simple Chart.js visualization for draw matching"""
        try:
            results = data.get('results', [])
            metadata = data.get('metadata', {})
            
            # Extract target from filename
            target = "Unknown"
            if 'draw_match_' in output_file:
                try:
                    target = output_file.split('draw_match_')[1].split('_')[0]
                except:
                    target = "555"
            
            # Calculate stats
            total_jobs = len(results)
            match_jobs = len([r for r in results if r.get('match_rate', 0) > 0])
            no_match_jobs = total_jobs - match_jobs
            avg_runtime = sum(r.get('elapsed', 0) for r in results) / max(1, total_jobs)
            
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Draw Match Results - Target {target}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        body {{ background: rgb(26,26,46); color: white; font-family: Arial; margin: 20px; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .chart-box {{ background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; }}
        .metrics {{ background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        canvas {{ max-height: 300px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Draw Match Analysis</h1>
            <h2>Target Number: {target}</h2>
        </div>
        
        <div class="metrics">
            <h3>Search Results</h3>
            <p>Jobs Executed: {total_jobs}</p>
            <p>Jobs with Matches: {match_jobs}</p>
            <p>Jobs without Matches: {no_match_jobs}</p>
            <p>Average Runtime: {avg_runtime:.2f}s</p>
        </div>
        
        <div class="charts">
            <div class="chart-box">
                <h3>Match Distribution</h3>
                <canvas id="matchChart"></canvas>
            </div>
            <div class="chart-box">
                <h3>Job Performance</h3>
                <canvas id="perfChart"></canvas>
            </div>
        </div>
    </div>
    
    <script>
        // Match distribution chart
        new Chart(document.getElementById('matchChart'), {{
            type: 'doughnut',
            data: {{
                labels: ['Jobs with Matches', 'Jobs without Matches'],
                datasets: [{{
                    data: [{match_jobs}, {no_match_jobs}],
                    backgroundColor: ['rgb(34,197,94)', 'rgb(239,68,68)']
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ labels: {{ color: 'white' }} }} }}
            }}
        }});
        
        // Performance chart
        new Chart(document.getElementById('perfChart'), {{
            type: 'bar',
            data: {{
                labels: {[f"Job {i+1}" for i in range(min(10, total_jobs))]},
                datasets: [{{
                    label: 'Runtime (seconds)',
                    data: {[r.get('elapsed', 0) for r in results[:10]]},
                    backgroundColor: 'rgb(99,102,241)'
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ labels: {{ color: 'white' }} }} }},
                scales: {{
                    y: {{ ticks: {{ color: 'white' }} }},
                    x: {{ ticks: {{ color: 'white' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(html)
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def compare_multiple_results(self):
        """Compare multiple analysis results"""
        print("Compare Multiple Results")
        print("=" * 30)
        
        results_dir = "results"
        if not os.path.exists(results_dir):
            print("No results directory found")
            input("Press Enter to continue...")
            return
        
        json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        if len(json_files) < 2:
            print("Need at least 2 result files for comparison")
        print("=" * 30)

        results_dir = "results"
        if not os.path.exists(results_dir):
            print("No results directory found")
            input("Press Enter to continue...")
            return

        json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        if len(json_files) < 2:
            print("Need at least 2 result files for comparison")
            input("Press Enter to continue...")
            return

        print("Available files:")
        for i, filename in enumerate(json_files[-15:], 1):
            print(f"  {i:2d}. {filename}")

        try:
            files_input = input("Enter file numbers to compare (e.g., 1,3,5): ")
            indices = [int(x.strip()) - 1 for x in files_input.split(',')]

            comparison_data = []
            for idx in indices:
                if 0 <= idx < len(json_files[-15:]):
                    filename = json_files[-15:][idx]
                    filepath = os.path.join(results_dir, filename)

                    with open(filepath, 'r') as f:
                        data = json.load(f)

                    data['filename'] = filename
                    comparison_data.append(data)

            if len(comparison_data) >= 2:
                print(f"\nComparing {len(comparison_data)} files...")
                self.display_comparison_table(comparison_data)

                if input("Create comparison visualization? (y/N): ").lower() == 'y':
                    html_file = self.create_comparison_html(comparison_data)
                    self.open_in_browser(html_file)
            else:
                print("Invalid file selection")

        except Exception as e:
            print(f"Error: {e}")

        input("Press Enter to continue...")

    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\nGenerate Comprehensive Report")
        print("=" * 30)

        results_dir = "results"
        if not os.path.exists(results_dir):
            print("No results directory found")
            input("Press Enter to continue...")
            return

        json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        if not json_files:
            print("No result files found")
            input("Press Enter to continue...")
            return

        print(f"Found {len(json_files)} result files")
        print("Generating comprehensive report...")

        # Analyze all files
        all_data = []
        for filename in json_files:
            try:
                filepath = os.path.join(results_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                data['filename'] = filename
                all_data.append(data)
            except:
                continue

        if all_data:
            report_data = self.generate_report_data(all_data)
            html_file = self.create_report_html(report_data)

            print(f"Report generated: {html_file}")
            self.open_in_browser(html_file)
        else:
            print("No valid data found for report")

        input("Press Enter to continue...")

    def live_cluster_monitor(self):
        """Real-time cluster monitoring with actual GPU data"""
        print("\nLive Cluster Monitor")
        print("=" * 30)
        print("Creating real-time cluster monitor with actual GPU data...")

        # Get real GPU data
        gpu_data = self.gpu_collector.get_all_gpu_data()
        cluster_summary = self.gpu_collector.get_cluster_summary()

        html_file = self.create_live_monitor_html(gpu_data, cluster_summary)

        print("Live cluster monitor opened successfully")
        print(f"Monitoring {cluster_summary['total_gpus']} GPUs")
        print(f"Active GPUs: {cluster_summary['active_gpus']}")
        print(f"Average utilization: {cluster_summary['avg_utilization']}%")

        self.open_in_browser(html_file)
        input("Press Enter to continue...")

    def gpu_performance_dashboard(self):
        """Detailed GPU performance dashboard with real metrics"""
        print("\nGPU Performance Dashboard")
        print("=" * 30)
        print("Creating detailed GPU performance dashboard...")

        # Get comprehensive GPU data
        gpu_data = self.gpu_collector.get_all_gpu_data()
        cluster_summary = self.gpu_collector.get_cluster_summary()

        html_file = self.create_gpu_dashboard_html(gpu_data, cluster_summary)

        print("GPU performance dashboard opened successfully")
        print(f"Total Power Draw: {cluster_summary['total_power']:.1f}W")
        print(f"Memory Usage: {cluster_summary['memory_utilization']:.1f}%")
        print(f"Average Temperature: {cluster_summary['avg_temperature']:.1f}°C")

        self.open_in_browser(html_file)
        input("Press Enter to continue...")

    def network_status_monitor(self):
        """Network connectivity and status monitoring"""
        print("\nNetwork Status Monitor")
        print("=" * 30)
        print("Checking network connectivity...")

        # Test connectivity to nodes
        connectivity = self.test_node_connectivity()
        html_file = self.create_network_status_html(connectivity)

        print("Network status monitor opened successfully")
        self.open_in_browser(html_file)
        input("Press Enter to continue...")

    def create_custom_visualization(self):
        """Create custom visualization dashboards"""
        print("\nCreate Custom Visualization")
        print("=" * 30)
        print("Custom dashboard types:")
        print("1. Performance Timeline")
        print("2. GPU Comparison Chart")
        print("3. System Health Overview")
        print("4. Analysis Results Summary")

        choice = input("Select dashboard type (1-4): ")

        if choice == '1':
            html_file = self.create_performance_timeline()
        elif choice == '2':
            html_file = self.create_gpu_comparison()
        elif choice == '3':
            html_file = self.create_health_overview()
        elif choice == '4':
            html_file = self.create_results_summary()
        else:
            print("Invalid choice")
            input("Press Enter to continue...")
            return

        print("Custom visualization created successfully")
        self.open_in_browser(html_file)
        input("Press Enter to continue...")

    def export_graphs(self):
        """Export visualization data and graphs"""
        print("\nExport Graphs")
        print("=" * 30)
        print("Export options:")
        print("1. Current GPU data (CSV)")
        print("2. Analysis results (JSON)")
        print("3. System metrics (CSV)")
        print("4. Performance history (JSON)")

        choice = input("Select export type (1-4): ")
        timestamp = int(time.time())

        try:
            if choice == '1':
                gpu_data = self.gpu_collector.get_all_gpu_data()
                filename = f"gpu_data_export_{timestamp}.csv"
                self.export_gpu_data_csv(gpu_data, filename)
                print(f"GPU data exported to: {filename}")

            elif choice == '2':
                filename = f"results_export_{timestamp}.json"
                self.export_results_json(filename)
                print(f"Results exported to: {filename}")

            elif choice == '3':
                cluster_data = self.gpu_collector.get_cluster_summary()
                filename = f"system_metrics_{timestamp}.csv"
                self.export_system_metrics_csv(cluster_data, filename)
                print(f"System metrics exported to: {filename}")

            elif choice == '4':
                filename = f"performance_history_{timestamp}.json"
                self.export_performance_history(filename)
                print(f"Performance history exported to: {filename}")

        except Exception as e:
            print(f"Export failed: {e}")

        input("Press Enter to continue...")

    def system_requirements_check(self):
        """Check system requirements for visualization"""
        print("\nSystem Requirements Check")
        print("=" * 35)

        # Check browser
        browsers = ['firefox', 'google-chrome', 'chromium-browser', 'chrome']
        browser_found = None
        for browser in browsers:
            try:
                subprocess.run(['which', browser], capture_output=True, check=True)
                browser_found = browser
                break
            except:
                continue

        if browser_found:
            print(f"✅ Browser found: {browser_found}")
        else:
            print("❌ No supported browser found")

        # Check HTML template
        template_path = "prng_visualizer.html"
        if os.path.exists(template_path):
            print("✅ HTML template: prng_visualizer.html")
        else:
            print("❌ HTML template missing")

        # Check results directory
        results_dir = "results"
        if os.path.exists(results_dir):
            json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
            print(f"✅ Results directory: {len(json_files)} JSON files found")
        else:
            print("❌ Results directory not found")

        # Check GPU data collection
        try:
            gpu_data = self.gpu_collector.get_all_gpu_data()
            print(f"✅ GPU monitoring: {len(gpu_data)} GPUs detected")
        except Exception as e:
            print(f"❌ GPU monitoring: {e}")

        print("\nVisualization system is ready!")
        input("Press Enter to continue...")

    def create_live_monitor_html(self, gpu_data: List[Dict], cluster_summary: Dict) -> str:
        """Create HTML for live cluster monitoring with real data"""
        timestamp = int(time.time())
        filename = f"live_cluster_monitor_{timestamp}.html"
        filepath = os.path.join(self.output_dir, filename)

        gpu_data_json = json.dumps(gpu_data, default=str)
        summary_json = json.dumps(cluster_summary, default=str)

        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Live Cluster Monitor - Real GPU Data</title>
    <style>
        body {{ font-family: 'Courier New', monospace; background: #000; color: #0f0; margin: 0; padding: 20px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .summary {{ display: flex; justify-content: space-around; margin-bottom: 30px; }}
        .metric {{ text-align: center; padding: 10px; border: 1px solid #0f0; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .nodes {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .node {{ border: 2px solid #0f0; border-radius: 10px; padding: 15px; margin-bottom: 20px; min-width: 300px; }}
        .node-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; text-align: center; }}
        .gpu-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; }}
        .gpu-card {{ border: 1px solid #0f0; padding: 8px; border-radius: 5px; text-align: center; font-size: 12px; }}
        .gpu-active {{ background: rgba(0, 255, 0, 0.1); }}
        .gpu-idle {{ background: rgba(0, 255, 0, 0.05); }}
        .gpu-offline {{ background: rgba(255, 0, 0, 0.1); border-color: #f00; color: #f00; }}
        .refresh-btn {{ position: fixed; top: 20px; right: 20px; padding: 10px; background: #0f0; color: #000; border: none; border-radius: 5px; cursor: pointer; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>LIVE CLUSTER MONITOR - 26 GPU SYSTEM</h1>
        <p>Real-time hardware monitoring - Updated: <span id="updateTime">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span></p>
    </div>

    <div class="summary">
        <div class="metric">
            <div class="metric-value" id="totalGPUs">{cluster_summary['total_gpus']}</div>
            <div>Total GPUs</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="activeGPUs">{cluster_summary['active_gpus']}</div>
            <div>Active GPUs</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="avgUtil">{cluster_summary['avg_utilization']}%</div>
            <div>Avg Utilization</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="avgTemp">{cluster_summary['avg_temperature']}°C</div>
            <div>Avg Temperature</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="totalPower">{cluster_summary['total_power']}W</div>
            <div>Total Power</div>
        </div>
    </div>

    <div class="nodes">
        <div class="node">
            <div class="node-title">ZEUS (localhost) - 2x RTX 3080 Ti</div>
            <div class="gpu-grid" id="zeusGrid"></div>
        </div>

        <div class="node">
            <div class="node-title">RIG-6600 (192.168.3.120) - 12x RX 6600</div>
            <div class="gpu-grid" id="rig6600Grid"></div>
        </div>

        <div class="node">
            <div class="node-title">RIG-6600B (192.168.3.154) - 12x RX 6600</div>
            <div class="gpu-grid" id="rig6600bGrid"></div>
        </div>
    </div>

    <button class="refresh-btn" onclick="location.reload()">REFRESH</button>

    <script>
        const gpuData = {gpu_data_json};
        const clusterSummary = {summary_json};

        function createGPUCard(gpu) {{
            const card = document.createElement('div');
            card.className = `gpu-card gpu-${{gpu.status}}`;
            card.innerHTML = `
                <div><strong>${{gpu.name}}</strong></div>
                <div>Util: ${{gpu.utilization}}%</div>
                <div>Temp: ${{gpu.temperature}}°C</div>
                <div>Mem: ${{(gpu.memory_used/1024).toFixed(1)}}GB</div>
                <div>Power: ${{gpu.power.toFixed(1)}}W</div>
            `;
            return card;
        }}

        function initializeGPUs() {{
            document.getElementById('zeusGrid').innerHTML = '';
            document.getElementById('rig6600Grid').innerHTML = '';
            document.getElementById('rig6600bGrid').innerHTML = '';

            gpuData.forEach(gpu => {{
                const card = createGPUCard(gpu);

                if (gpu.node === 'localhost') {{
                    document.getElementById('zeusGrid').appendChild(card);
                }} else if (gpu.node === '192.168.3.120') {{
                    document.getElementById('rig6600Grid').appendChild(card);
                }} else if (gpu.node === '192.168.3.154') {{
                    document.getElementById('rig6600bGrid').appendChild(card);
                }}
            }});
        }}

        initializeGPUs();

        setInterval(() => {{
            location.reload();
        }}, 30000);
    </script>
</body>
</html>'''

        with open(filepath, 'w') as f:
            f.write(html_content)

        return filepath

    def create_gpu_dashboard_html(self, gpu_data: List[Dict], cluster_summary: Dict) -> str:
        """Create comprehensive GPU performance dashboard"""
        timestamp = int(time.time())
        filename = f"gpu_performance_dashboard_{timestamp}.html"
        filepath = os.path.join(self.output_dir, filename)

        # Prepare data for charts
        utilizations = [gpu['utilization'] for gpu in gpu_data]
        temperatures = [gpu['temperature'] for gpu in gpu_data]
        power_usage = [gpu['power'] for gpu in gpu_data]
        memory_usage = [(gpu['memory_used'] / gpu['memory_total']) * 100 if gpu['memory_total'] > 0 else 0 for gpu in gpu_data]

        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>GPU Performance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; margin: 0; padding: 20px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
        .chart-container {{ background: #2a2a2a; padding: 20px; border-radius: 10px; height: 400px; }}
        .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .metric {{ text-align: center; padding: 15px; background: #333; border-radius: 10px; }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #4CAF50; }}
        .metric-label {{ font-size: 14px; color: #ccc; }}
        .gpu-table {{ width: 100%; background: #2a2a2a; border-radius: 10px; padding: 20px; margin-top: 20px; }}
        .gpu-table table {{ width: 100%; border-collapse: collapse; }}
        .gpu-table th, .gpu-table td {{ padding: 10px; text-align: left; border-bottom: 1px solid #444; }}
        .gpu-table th {{ background: #333; }}
        .status-active {{ color: #4CAF50; }}
        .status-idle {{ color: #FFC107; }}
        .status-offline {{ color: #F44336; }}
        canvas {{ max-height: 350px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>GPU PERFORMANCE DASHBOARD</h1>
        <p>Real-time monitoring of 26-GPU cluster performance</p>
        <p>Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <div class="metrics">
        <div class="metric">
            <div class="metric-value">{cluster_summary['total_gpus']}</div>
            <div class="metric-label">Total GPUs</div>
        </div>
        <div class="metric">
            <div class="metric-value">{cluster_summary['avg_utilization']}%</div>
            <div class="metric-label">Avg Utilization</div>
        </div>
        <div class="metric">
            <div class="metric-value">{cluster_summary['avg_temperature']}°C</div>
            <div class="metric-label">Avg Temperature</div>
        </div>
        <div class="metric">
            <div class="metric-value">{cluster_summary['total_power']:.0f}W</div>
            <div class="metric-label">Total Power</div>
        </div>
        <div class="metric">
            <div class="metric-value">{cluster_summary['memory_utilization']:.1f}%</div>
            <div class="metric-label">Memory Usage</div>
        </div>
    </div>

    <div class="dashboard">
        <div class="chart-container">
            <h3>GPU Utilization (%)</h3>
            <canvas id="utilizationChart"></canvas>
        </div>
        <div class="chart-container">
            <h3>GPU Temperature (°C)</h3>
            <canvas id="temperatureChart"></canvas>
        </div>
        <div class="chart-container">
            <h3>Power Consumption (W)</h3>
            <canvas id="powerChart"></canvas>
        </div>
        <div class="chart-container">
            <h3>Memory Usage (%)</h3>
            <canvas id="memoryChart"></canvas>
        </div>
    </div>

    <div class="gpu-table">
        <h3>Individual GPU Status - All 26 GPUs</h3>
        <table>
            <thead>
                <tr>
                    <th>GPU ID</th>
                    <th>Name</th>
                    <th>Node</th>
                    <th>Status</th>
                    <th>Utilization</th>
                    <th>Temperature</th>
                    <th>Memory</th>
                    <th>Power</th>
                </tr>
            </thead>
            <tbody>'''

        for gpu in gpu_data:
            status_class = f"status-{gpu['status']}"
            memory_gb = f"{gpu['memory_used']/1024:.1f}/{gpu['memory_total']/1024:.1f}GB"
            html_content += f'''
                <tr>
                    <td>{gpu['id']}</td>
                    <td>{gpu['name']}</td>
                    <td>{gpu['node']}</td>
                    <td class="{status_class}">{gpu['status'].upper()}</td>
                    <td>{gpu['utilization']}%</td>
                    <td>{gpu['temperature']}°C</td>
                    <td>{memory_gb}</td>
                    <td>{gpu['power']:.1f}W</td>
                </tr>'''

        html_content += f'''
            </tbody>
        </table>
    </div>

    <script>
        // GPU Data
        const gpuLabels = {json.dumps([f"GPU{gpu['id']}" for gpu in gpu_data])};
        const utilizationData = {utilizations};
        const temperatureData = {temperatures};
        const powerData = {power_usage};
        const memoryData = {memory_usage};

        // Chart colors
        const colors = {{
            utilization: 'rgba(76, 175, 80, 0.8)',
            temperature: 'rgba(255, 152, 0, 0.8)',
            power: 'rgba(244, 67, 54, 0.8)',
            memory: 'rgba(33, 150, 243, 0.8)'
        }};

        // Chart options
        const chartOptions = {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{ display: false }}
            }},
            scales: {{
                y: {{ beginAtZero: true }}
            }}
        }};

        // Utilization Chart
        new Chart(document.getElementById('utilizationChart'), {{
            type: 'bar',
            data: {{
                labels: gpuLabels,
                datasets: [{{
                    data: utilizationData,
                    backgroundColor: colors.utilization,
                    borderColor: colors.utilization,
                    borderWidth: 1
                }}]
            }},
            options: {{
                ...chartOptions,
                scales: {{
                    y: {{ max: 100, beginAtZero: true }}
                }}
            }}
        }});

        // Temperature Chart
        new Chart(document.getElementById('temperatureChart'), {{
            type: 'line',
            data: {{
                labels: gpuLabels,
                datasets: [{{
                    data: temperatureData,
                    backgroundColor: 'rgba(255, 152, 0, 0.2)',
                    borderColor: colors.temperature,
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1
                }}]
            }},
            options: {{
                ...chartOptions,
                scales: {{
                    y: {{ max: 90, beginAtZero: true }}
                }}
            }}
        }});

        // Power Chart
        new Chart(document.getElementById('powerChart'), {{
            type: 'bar',
            data: {{
                labels: gpuLabels,
                datasets: [{{
                    data: powerData,
                    backgroundColor: colors.power,
                    borderColor: colors.power,
                    borderWidth: 1
                }}]
            }},
            options: chartOptions
        }});

        // Memory Chart (simplified bar chart instead of doughnut for better visibility)
        new Chart(document.getElementById('memoryChart'), {{
            type: 'bar',
            data: {{
                labels: gpuLabels,
                datasets: [{{
                    data: memoryData,
                    backgroundColor: colors.memory,
                    borderColor: colors.memory,
                    borderWidth: 1
                }}]
            }},
            options: {{
                ...chartOptions,
                scales: {{
                    y: {{ max: 100, beginAtZero: true }}
                }}
            }}
        }});

        // Auto refresh every 60 seconds
        setTimeout(() => {{
            location.reload();
        }}, 60000);

        console.log('GPU Dashboard loaded with', gpuLabels.length, 'GPUs');
    </script>
</body>
</html>'''

        with open(filepath, 'w') as f:
            f.write(html_content)

        return filepath

    def create_network_status_html(self, connectivity: Dict) -> str:
        """Create network status monitoring HTML"""
        timestamp = int(time.time())
        filename = f"network_status_{timestamp}.html"
        filepath = os.path.join(self.output_dir, filename)

        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Network Status Monitor</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; margin: 20px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .node {{ background: #2a2a2a; padding: 20px; margin: 10px; border-radius: 10px; }}
        .status-online {{ border-left: 5px solid #4CAF50; }}
        .status-offline {{ border-left: 5px solid #F44336; }}
        .latency {{ color: #ccc; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>NETWORK STATUS MONITOR</h1>
        <p>Cluster connectivity status</p>
    </div>'''

        for host, status in connectivity.items():
            status_class = f"status-{status['status']}"
            html_content += f'''
    <div class="node {status_class}">
        <h3>{host}</h3>
        <p>Status: {status['status'].upper()}</p>
        <p class="latency">Latency: {status['latency']:.1f}ms</p>
    </div>'''

        html_content += '''
</body>
</html>'''

        with open(filepath, 'w') as f:
            f.write(html_content)

        return filepath

    def test_node_connectivity(self) -> Dict:
        """Test connectivity to all cluster nodes"""
        connectivity = {
            'localhost': {'status': 'online', 'latency': 0},
            '192.168.3.120': {'status': 'unknown', 'latency': 0},
            '192.168.3.154': {'status': 'unknown', 'latency': 0}
        }

        for host in ['192.168.3.120', '192.168.3.154']:
            try:
                result = subprocess.run(['ping', '-c', '1', '-W', '2', host],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    output = result.stdout
                    if 'time=' in output:
                        latency_str = output.split('time=')[1].split(' ')[0]
                        latency = float(latency_str)
                        connectivity[host] = {'status': 'online', 'latency': latency}
                    else:
                        connectivity[host] = {'status': 'online', 'latency': 0}
                else:
                    connectivity[host] = {'status': 'offline', 'latency': 0}
            except:
                connectivity[host] = {'status': 'offline', 'latency': 0}

        return connectivity

    def open_in_browser(self, filepath: str):
        """Open HTML file in browser"""
        try:
            browsers = ['xdg-open', 'firefox', 'google-chrome', 'chromium-browser']
            for browser in browsers:
                try:
                    subprocess.run([browser, filepath], check=True)
                    return
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            print(f"Could not open browser. File saved to: {filepath}")
        except Exception as e:
            print(f"Error opening browser: {e}")

    def display_result_summary(self, data: Dict):
        """Display summary of analysis result"""
        if 'metadata' in data:
            meta = data['metadata']
            print(f"Analysis completed: {meta.get('successful_jobs', 0)}/{meta.get('total_jobs', 0)} jobs")
            print(f"Runtime: {meta.get('total_runtime', 0):.1f} seconds")
            print(f"GPUs used: {meta.get('total_gpus', 0)}")

        results = data.get('results', [])
        print(f"Matches found: {len(results)}")

    def display_detailed_analysis(self, data: Dict):
        """Display detailed analysis information"""
        self.display_result_summary(data)

        if 'metadata' in data:
            meta = data['metadata']
            print(f"\nDetailed metrics:")
            print(f"  Nodes used: {meta.get('nodes_used', 0)}")
            print(f"  Hardware optimized: {meta.get('hardware_optimized', False)}")

        if data.get('failed_jobs'):
            print(f"  Failed jobs: {len(data['failed_jobs'])}")

    def display_comparison_table(self, comparison_data: List[Dict]):
        """Display comparison table for multiple results"""
        print("\nComparison Results:")
        print("-" * 80)
        print(f"{'File':<30} {'Jobs':<10} {'Runtime':<10} {'Matches':<10}")
        print("-" * 80)

        for data in comparison_data:
            filename = data.get('filename', 'Unknown')[:28]
            meta = data.get('metadata', {})
            jobs = f"{meta.get('successful_jobs', 0)}/{meta.get('total_jobs', 0)}"
            runtime = f"{meta.get('total_runtime', 0):.1f}s"
            matches = len(data.get('results', []))

            print(f"{filename:<30} {jobs:<10} {runtime:<10} {matches:<10}")

    def create_result_html(self, data: Dict, filename: str) -> str:
        """Create basic HTML visualization for a result"""
        timestamp = int(time.time())
        html_filename = f"result_viz_{timestamp}.html"
        filepath = os.path.join(self.output_dir, html_filename)

        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Analysis Result: {filename}</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; margin: 20px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .summary {{ background: #2a2a2a; padding: 20px; border-radius: 10px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Analysis Result</h1>
        <h2>{filename}</h2>
    </div>
    <div class="summary">
        <h3>Summary</h3>
        <pre>{json.dumps(data, indent=2)[:1000]}...</pre>
    </div>
</body>
</html>'''

        with open(filepath, 'w') as f:
            f.write(html_content)

        return filepath

    def create_detailed_html(self, data: Dict, filename: str) -> str:
        """Create detailed HTML report for a result"""
        return self.create_result_html(data, filename)

    def create_comparison_html(self, comparison_data: List[Dict]) -> str:
        """Create comparison HTML visualization"""
        timestamp = int(time.time())
        html_filename = f"comparison_viz_{timestamp}.html"
        filepath = os.path.join(self.output_dir, html_filename)

        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Results Comparison</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; margin: 20px; }}
        .comparison {{ background: #2a2a2a; padding: 20px; border-radius: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>Results Comparison</h1>'''

        for i, data in enumerate(comparison_data):
            html_content += f'''
    <div class="comparison">
        <h3>File {i+1}: {data.get('filename', 'Unknown')}</h3>
        <p>Jobs: {data.get('metadata', {}).get('successful_jobs', 0)}</p>
        <p>Runtime: {data.get('metadata', {}).get('total_runtime', 0):.1f}s</p>
        <p>Matches: {len(data.get('results', []))}</p>
    </div>'''

        html_content += '''
</body>
</html>'''

        with open(filepath, 'w') as f:
            f.write(html_content)

        return filepath

    def generate_report_data(self, all_data: List[Dict]) -> Dict:
        """Generate comprehensive report data"""
        total_jobs = sum(d.get('metadata', {}).get('successful_jobs', 0) for d in all_data)
        total_runtime = sum(d.get('metadata', {}).get('total_runtime', 0) for d in all_data)
        total_matches = sum(len(d.get('results', [])) for d in all_data)

        return {
            'total_files': len(all_data),
            'total_jobs': total_jobs,
            'total_runtime': total_runtime,
            'total_matches': total_matches,
            'avg_runtime': total_runtime / len(all_data) if all_data else 0
        }

    def create_report_html(self, report_data: Dict) -> str:
        """Create comprehensive report HTML"""
        timestamp = int(time.time())
        html_filename = f"comprehensive_report_{timestamp}.html"
        filepath = os.path.join(self.output_dir, html_filename)

        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; margin: 20px; }}
        .metric {{ background: #2a2a2a; padding: 15px; margin: 10px; border-radius: 10px; }}
    </style>
</head>
<body>
    <h1>Comprehensive Analysis Report</h1>
    <div class="metric">
        <h3>Total Files Analyzed: {report_data['total_files']}</h3>
    </div>
    <div class="metric">
        <h3>Total Jobs Completed: {report_data['total_jobs']}</h3>
    </div>
    <div class="metric">
        <h3>Total Runtime: {report_data['total_runtime']:.1f} seconds</h3>
    </div>
    <div class="metric">
        <h3>Total Matches Found: {report_data['total_matches']}</h3>
    </div>
    <div class="metric">
        <h3>Average Runtime: {report_data['avg_runtime']:.1f} seconds</h3>
    </div>
</body>
</html>'''

        with open(filepath, 'w') as f:
            f.write(html_content)

        return filepath

    def create_performance_timeline(self) -> str:
        """Create performance timeline visualization"""
        timestamp = int(time.time())
        html_filename = f"performance_timeline_{timestamp}.html"
        filepath = os.path.join(self.output_dir, html_filename)

        html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Performance Timeline</title>
    <style>
        body { font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; margin: 20px; }
    </style>
</head>
<body>
    <h1>Performance Timeline</h1>
    <p>Timeline visualization would be implemented here</p>
</body>
</html>'''

        with open(filepath, 'w') as f:
            f.write(html_content)

        return filepath

    def create_gpu_comparison(self) -> str:
        """Create GPU comparison chart"""
        timestamp = int(time.time())
        html_filename = f"gpu_comparison_{timestamp}.html"
        filepath = os.path.join(self.output_dir, html_filename)

        html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>GPU Comparison</title>
    <style>
        body { font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; margin: 20px; }
    </style>
</head>
<body>
    <h1>GPU Comparison Chart</h1>
    <p>GPU comparison visualization would be implemented here</p>
</body>
</html>'''

        with open(filepath, 'w') as f:
            f.write(html_content)

        return filepath

    def create_health_overview(self) -> str:
        """Create system health overview"""
        timestamp = int(time.time())
        html_filename = f"health_overview_{timestamp}.html"
        filepath = os.path.join(self.output_dir, html_filename)

        html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>System Health Overview</title>
    <style>
        body { font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; margin: 20px; }
    </style>
</head>
<body>
    <h1>System Health Overview</h1>
    <p>Health overview visualization would be implemented here</p>
</body>
</html>'''

        with open(filepath, 'w') as f:
            f.write(html_content)

        return filepath

    def create_results_summary(self) -> str:
        """Create analysis results summary"""
        timestamp = int(time.time())
        html_filename = f"results_summary_{timestamp}.html"
        filepath = os.path.join(self.output_dir, html_filename)

        html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Analysis Results Summary</title>
    <style>
        body { font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; margin: 20px; }
    </style>
</head>
<body>
    <h1>Analysis Results Summary</h1>
    <p>Results summary visualization would be implemented here</p>
</body>
</html>'''

        with open(filepath, 'w') as f:
            f.write(html_content)

        return filepath

    def export_gpu_data_csv(self, gpu_data: List[Dict], filename: str):
        """Export GPU data to CSV"""
        with open(filename, 'w') as f:
            f.write("GPU_ID,Name,Node,Status,Utilization,Temperature,Memory_Used_MB,Memory_Total_MB,Power_W\n")
            for gpu in gpu_data:
                f.write(f"{gpu['id']},{gpu['name']},{gpu['node']},{gpu['status']},")
                f.write(f"{gpu['utilization']},{gpu['temperature']},{gpu['memory_used']},")
                f.write(f"{gpu['memory_total']},{gpu['power']}\n")

    def export_results_json(self, filename: str):
        """Export analysis results to JSON"""
        results_dir = "results"
        if os.path.exists(results_dir):
            combined_results = []
            for json_file in os.listdir(results_dir):
                if json_file.endswith('.json'):
                    try:
                        with open(os.path.join(results_dir, json_file), 'r') as f:
                            data = json.load(f)
                        data['source_file'] = json_file
                        combined_results.append(data)
                    except:
                        continue

            with open(filename, 'w') as f:
                json.dump(combined_results, f, indent=2, default=str)

    def export_system_metrics_csv(self, cluster_data: Dict, filename: str):
        """Export system metrics to CSV"""
        with open(filename, 'w') as f:
            f.write("Metric,Value\n")
            for key, value in cluster_data.items():
                f.write(f"{key},{value}\n")

    def export_performance_history(self, filename: str):
        """Export performance history to JSON"""
        history = getattr(self.core, 'performance_history', [])
        with open(filename, 'w') as f:
            json.dump(history, f, indent=2, default=str)

    def shutdown(self):
        """Cleanup visualization resources"""
        print(f"Shutting down {self.module_name}...")
        pass
    def create_draw_match_visualization_html(self, data, output_file):
        """Create HTML visualization for draw matching results"""
        try:
            # Extract draw matching specific data
            results = data.get('results', [])
            metadata = data.get('metadata', {})
            
            # Analyze draw matching results
            total_seeds = sum(result.get('n_seeds', 0) for result in results)
            target_matches = []
            search_statistics = {
                'total_jobs': len(results),
                'total_seeds_tested': total_seeds,
                'total_runtime': metadata.get('total_runtime', 0),
                'gpus_used': metadata.get('total_gpus', 0)
            }
            
            # Extract matching results
            for result in results:
                if 'matches' in result:
                    target_matches.extend(result['matches'])
                elif 'matching_seeds' in result:
                    target_matches.extend(result['matching_seeds'])
                elif 'target_matches' in result:
                    target_matches.extend(result['target_matches'])
            
            # Determine target number from filename
            target_number = "Unknown"
            if 'draw_match_' in output_file:
                try:
                    parts = output_file.split('draw_match_')[1].split('_')[0]
                    target_number = parts
                except:
                    pass
            
            # Create HTML content
            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Draw Match Analysis - Target {target_number}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, rgb(26,26,46), rgb(22,33,62));
            color: white;
            min-height: 100vh;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .card {{
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 10px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
        }}
        .metric-label {{
            font-weight: bold;
            color: rgb(74,222,128);
        }}
        .metric-value {{
            color: white;
        }}
        .matches-list {{
            max-height: 300px;
            overflow-y: auto;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            padding: 15px;
        }}
        .match-item {{
            padding: 8px;
            margin: 5px 0;
            background: rgba(34,197,94,0.2);
            border-left: 4px solid rgb(34,197,94);
            border-radius: 4px;
        }}
        .no-matches {{
            text-align: center;
            color: rgb(251,191,36);
            font-style: italic;
            padding: 20px;
        }}
        .chart-container {{
            position: relative;
            height: 300px;
            margin: 20px 0;
        }}
        h1, h2, h3 {{
            margin-top: 0;
        }}
        .target-highlight {{
            font-size: 2em;
            color: rgb(96,165,250);
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Draw Match Analysis Results</h1>
        <p>Target Number: <span class="target-highlight">{target_number}</span></p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="container">
        <div class="card">
            <h2>Search Summary</h2>
            <div class="metric">
                <span class="metric-label">Target Number:</span>
                <span class="metric-value">{target_number}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Seeds Tested:</span>
                <span class="metric-value">{search_statistics['total_seeds_tested']:,}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Jobs Executed:</span>
                <span class="metric-value">{search_statistics['total_jobs']}</span>
            </div>
            <div class="metric">
                <span class="metric-label">GPUs Used:</span>
                <span class="metric-value">{search_statistics['gpus_used']}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Runtime:</span>
                <span class="metric-value">{search_statistics['total_runtime']:.1f}s</span>
            </div>
            <div class="metric">
                <span class="metric-label">Matches Found:</span>
                <span class="metric-value" style="color: {'rgb(34,197,94)' if len(target_matches) > 0 else 'rgb(239,68,68)'}; font-weight: bold;">{len(target_matches)}</span>
            </div>
        </div>

        <div class="card">
            <h2>Performance Metrics</h2>
            <div class="chart-container">
                <canvas id="performanceChart"></canvas>
            </div>
        </div>

        <div class="card full-width">
            <h2>Matching Seeds</h2>
            <div class="matches-list">
                {self._generate_matches_html(target_matches)}
            </div>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('performanceChart').getContext('2d');
        new Chart(ctx, {{
            type: 'doughnut',
            data: {{
                labels: ['Seeds Tested', 'Matches Found'],
                datasets: [{{
                    data: [{search_statistics['total_seeds_tested']}, {len(target_matches)}],
                    backgroundColor: [
                        'rgba(99, 102, 241, 0.8)',
                        'rgba(34, 197, 94, 0.8)'
                    ],
                    borderColor: [
                        'rgb(99, 102, 241)',
                        'rgb(34, 197, 94)'
                    ],
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        labels: {{
                            color: 'white'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
            
            # Write HTML file
            with open(output_file, 'w') as f:
                f.write(html_content)
            
            return True
            
        except Exception as e:
            print(f"Error creating draw match visualization: {e}")
            return False

    def _generate_matches_html(self, matches):
        """Generate HTML for the matches list"""
        if not matches:
            return '<div class="no-matches">No matching seeds found for this target number.</div>'
        
        html = ""
        for i, match in enumerate(matches[:50]):  # Limit to first 50 matches
            if isinstance(match, dict):
                seed = match.get('seed', 'Unknown')
                details = match.get('details', '')
            else:
                seed = match
                details = ''
            
            html += f'<div class="match-item">Match #{i+1}: Seed {seed} {details}</div>'
        
        if len(matches) > 50:
            html += f'<div class="no-matches">... and {len(matches) - 50} more matches</div>'
        
        return html

    def create_draw_match_visualization_html(self, data, output_file):
        """Create HTML visualization for draw matching results"""
        try:
            # Extract draw matching specific data
            results = data.get('results', [])
            metadata = data.get('metadata', {})
            
            # Analyze draw matching results
            total_seeds = sum(result.get('n_seeds', 0) for result in results)
            target_matches = []
            search_statistics = {
                'total_jobs': len(results),
                'total_seeds_tested': total_seeds,
                'total_runtime': metadata.get('total_runtime', 0),
                'gpus_used': metadata.get('total_gpus', 0)
            }
            
            # Extract matching results
            for result in results:
                if 'matches' in result:
                    target_matches.extend(result['matches'])
                elif 'matching_seeds' in result:
                    target_matches.extend(result['matching_seeds'])
                elif 'target_matches' in result:
                    target_matches.extend(result['target_matches'])
            
            # Determine target number from filename
            target_number = "Unknown"
            if 'draw_match_' in output_file:
                try:
                    parts = output_file.split('draw_match_')[1].split('_')[0]
                    target_number = parts
                except:
                    pass
            
            # Create HTML content
            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Draw Match Analysis - Target {target_number}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, rgb(26,26,46), rgb(22,33,62));
            color: white;
            min-height: 100vh;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .card {{
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 10px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
        }}
        .metric-label {{
            font-weight: bold;
            color: rgb(74,222,128);
        }}
        .metric-value {{
            color: white;
        }}
        .matches-list {{
            max-height: 300px;
            overflow-y: auto;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            padding: 15px;
        }}
        .match-item {{
            padding: 8px;
            margin: 5px 0;
            background: rgba(34,197,94,0.2);
            border-left: 4px solid rgb(34,197,94);
            border-radius: 4px;
        }}
        .no-matches {{
            text-align: center;
            color: rgb(251,191,36);
            font-style: italic;
            padding: 20px;
        }}
        .chart-container {{
            position: relative;
            height: 300px;
            margin: 20px 0;
        }}
        h1, h2, h3 {{
            margin-top: 0;
        }}
        .target-highlight {{
            font-size: 2em;
            color: rgb(96,165,250);
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Draw Match Analysis Results</h1>
        <p>Target Number: <span class="target-highlight">{target_number}</span></p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="container">
        <div class="card">
            <h2>Search Summary</h2>
            <div class="metric">
                <span class="metric-label">Target Number:</span>
                <span class="metric-value">{target_number}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Seeds Tested:</span>
                <span class="metric-value">{search_statistics['total_seeds_tested']:,}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Jobs Executed:</span>
                <span class="metric-value">{search_statistics['total_jobs']}</span>
            </div>
            <div class="metric">
                <span class="metric-label">GPUs Used:</span>
                <span class="metric-value">{search_statistics['gpus_used']}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Runtime:</span>
                <span class="metric-value">{search_statistics['total_runtime']:.1f}s</span>
            </div>
            <div class="metric">
                <span class="metric-label">Matches Found:</span>
                <span class="metric-value" style="color: {'rgb(34,197,94)' if len(target_matches) > 0 else 'rgb(239,68,68)'}; font-weight: bold;">{len(target_matches)}</span>
            </div>
        </div>

        <div class="card">
            <h2>Performance Metrics</h2>
            <div class="chart-container">
                <canvas id="performanceChart"></canvas>
            </div>
        </div>

        <div class="card full-width">
            <h2>Matching Seeds</h2>
            <div class="matches-list">
                {self._generate_matches_html(target_matches)}
            </div>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('performanceChart').getContext('2d');
        new Chart(ctx, {{
            type: 'doughnut',
            data: {{
                labels: ['Seeds Tested', 'Matches Found'],
                datasets: [{{
                    data: [{search_statistics['total_seeds_tested']}, {len(target_matches)}],
                    backgroundColor: [
                        'rgba(99, 102, 241, 0.8)',
                        'rgba(34, 197, 94, 0.8)'
                    ],
                    borderColor: [
                        'rgb(99, 102, 241)',
                        'rgb(34, 197, 94)'
                    ],
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        labels: {{
                            color: 'white'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
            
            # Write HTML file
            with open(output_file, 'w') as f:
                f.write(html_content)
            
            return True
            
        except Exception as e:
            print(f"Error creating draw match visualization: {e}")
            return False

    def _generate_matches_html(self, matches):
        """Generate HTML for the matches list"""
        if not matches:
            return '<div class="no-matches">No matching seeds found for this target number.</div>'
        
        html = ""
        for i, match in enumerate(matches[:50]):  # Limit to first 50 matches
            if isinstance(match, dict):
                seed = match.get('seed', 'Unknown')
                details = match.get('details', '')
            else:
                seed = match
                details = ''
            
            html += f'<div class="match-item">Match #{i+1}: Seed {seed} {details}</div>'
        
        if len(matches) > 50:
            html += f'<div class="no-matches">... and {len(matches) - 50} more matches</div>'
        
        return html

