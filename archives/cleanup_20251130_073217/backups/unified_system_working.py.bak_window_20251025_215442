#!/usr/bin/env python3
"""
Unified System Interface for Distributed PRNG Analysis - Main Entry Point
Modular architecture with compartmentalized features and working visualizations

Version: 4.1.0 (Modular + Visualizations)
Last Modified: September 14, 2025
Author: Development Team
Dependencies: All module files in modules/ directory
Status: Production ready - modular design with real-time GPU monitoring
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any

# Add modules directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

# Import all feature modules
from system_core import SystemCore
from modules.direct_analysis import DirectAnalysis
from modules.advanced_research import AdvancedResearch
from modules.database_manager import DatabaseManager
from modules.result_viewer import ResultViewer
from modules.system_monitor import SystemMonitor
from modules.file_manager import FileManager
from modules.performance_analytics import PerformanceAnalytics
from modules.visualization_manager import VisualizationManager

class UnifiedSystem:
    """
    Main unified system class that coordinates all feature modules

    Enhanced version with working GPU visualizations and all advanced features
    preserved from the working system.
    """

    def __init__(self):
        """Initialize the unified system with all feature modules"""
        print("Initializing Unified PRNG Analysis System...")
        print("=" * 60)

        # Initialize core system first
        self.core = SystemCore()

        # Initialize all feature modules with core reference
        self.modules = {
            'direct_analysis': DirectAnalysis(self.core),
            'advanced_research': AdvancedResearch(self.core),
            'database_manager': DatabaseManager(self.core),
            'result_viewer': ResultViewer(self.core),
            'system_monitor': SystemMonitor(self.core),
            'file_manager': FileManager(self.core),
            'performance_analytics': PerformanceAnalytics(self.core),
            'visualization_manager': VisualizationManager(self.core)
        }

        print("All modules initialized successfully")
        print("System ready with 26-GPU cluster visualization")
        print("=" * 60)

    def main_menu(self):
        """Main menu interface with modular navigation and visualizations"""
        while True:
            self.core.clear_screen()
            self.core.print_header()

            # Display system status
            self.core.display_system_status()

            print("\nMAIN MENU")
            print("-" * 30)
            print("ANALYSIS & EXECUTION:")
            print("  1. Direct Cluster Analysis")      # DirectAnalysis module
            print("  2. Advanced Research Jobs")       # AdvancedResearch module
            print("  3. Process DB Jobs")              # DatabaseManager module

            print("\nMONITORING & MANAGEMENT:")
            print("  4. View Job Queue")               # DatabaseManager module
            print("  5. Check Results")                # ResultViewer module
            print("  6. System Status")                # SystemMonitor module
            print("  7. Database Management")          # DatabaseManager module
            print("  8. Data Visualization & Monitoring") # VisualizationManager module

            print("\nMAINTENANCE & UTILITIES:")
            print("  9. File Management")              # FileManager module
            print(" 10. Performance Analytics")        # PerformanceAnalytics module
            print(" 11. System Diagnostics")           # SystemMonitor module
            print(" 12. Web Visualization Server")     # Web server for visualizations
            print(" 13. Exit")                         # Clean shutdown
            print("-" * 30)

            choice = input("Select option (1-13): ").strip()

            # Increment operation counter
            self.core.operation_count += 1

            # Route to appropriate module
            try:
                if choice == '1':
                    self.modules['direct_analysis'].menu()
                elif choice == '2':
                    self.modules['advanced_research'].menu()
                elif choice == '3':
                    self.modules['database_manager'].process_jobs_menu()
                elif choice == '4':
                    self.modules['database_manager'].browse_search_jobs()
                elif choice == '5':
                    self.modules['result_viewer'].menu()
                elif choice == '6':
                    self.modules['system_monitor'].status_menu()
                elif choice == '7':
                    self.modules['database_manager'].menu()
                elif choice == '8':
                    self.modules['visualization_manager'].menu()
                elif choice == '9':
                    self.modules['file_manager'].menu()
                elif choice == '10':
                    self.modules['performance_analytics'].menu()
                elif choice == '11':
                    self.modules['system_monitor'].diagnostics_menu()
                elif choice == '12':
                    self.web_server_menu()
               
                elif choice == '13':
                    self.shutdown()
                    break
                else:
                    print("Invalid choice. Press Enter to continue...")
                    input()

            except Exception as e:
                print(f"Error in module execution: {e}")
                self.core.log_operation("module_error", "ERROR", str(e))
                input("Press Enter to continue...")

    def start_web_server(self):
        """Start visualization web server for remote access"""
        print("\nWeb Visualization Server")
        print("=" * 40)

        viz_dir = "visualizations"
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        print("Starting web server on port 8000...")
        print(f"Access visualizations at: http://192.168.3.127:8000/")

        # List available visualizations
        html_files = [f for f in os.listdir(viz_dir) if f.endswith('.html')]
        if html_files:
            print("\nAvailable visualizations:")
            for file in html_files[-10:]:  # Show last 10 files
                print(f"  http://192.168.3.127:8000/{file}")
        else:
            print("No visualizations found. Use option 8 to create GPU dashboards first.")

        print("\nStarting server...")
        try:
            current_dir = os.getcwd()
            os.chdir(viz_dir)
            os.system("python3 -m http.server 8000 &")
            os.chdir(current_dir)
            print("Web server started successfully")
            print("Server runs in background - access dashboards via browser")
        except Exception as e:
            print(f"Error starting server: {e}")

        input("Press Enter to continue...")


    def web_server_menu(self):
        """Web server control menu"""
        while True:
            self.core.clear_screen()
            self.core.print_header()
            print("\nWEB VISUALIZATION SERVER")
            print("=" * 40)
            
            status = self.check_web_server_status()
            print(f"Status: {status}")
            
            if status == "Running":
                print(f"URL: http://192.168.3.127:8000/")
                viz_count = len(self.list_visualization_files())
                print(f"Visualizations: {viz_count} files")
            
            print("\nCONTROLS:")
            print("  1. Start Web Server")
            print("  2. Stop Web Server")
            print("  3. Server Status")
            print("  4. List Visualizations")
            print("  5. Delete Old Visualizations")
            print("  6. Delete All Visualizations")
            print("  7. Back to Main Menu")
            print("=" * 40)
            
            choice = input("Select option (1-7): ").strip()
            
            if choice == '1':
                self.start_web_server()
            elif choice == '2':
                self.stop_web_server()
            elif choice == '3':
                self.show_web_server_status()
            elif choice == '4':
                self.list_visualizations()
            elif choice == '5':
                self.delete_old_visualizations()
            elif choice == '6':
                self.delete_all_visualizations()
            elif choice == '7':
                break
            else:
                print("Invalid choice")
                input("Press Enter to continue...")
    
    def check_web_server_status(self):
        """Check if web server is running on port 8000"""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(('localhost', 8000))
                return "Running" if result == 0 else "Stopped"
        except:
            return "Unknown"
    
    def start_web_server(self):
        """Start the web server"""
        print("\nStarting Web Visualization Server...")
        
        if self.check_web_server_status() == "Running":
            print("Server is already running!")
            print("URL: http://192.168.3.127:8000/")
            input("Press Enter to continue...")
            return
        
        try:
            import subprocess
            import os
            
            # Ensure visualizations directory exists
            os.makedirs("visualizations", exist_ok=True)
            
            # Start server
            cmd = ['python3', '-m', 'http.server', '8000', '--directory', 'visualizations']
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Give it time to start
            import time
            time.sleep(2)
            
            if self.check_web_server_status() == "Running":
                print("✓ Web server started successfully!")
                print("URL: http://192.168.3.127:8000/")
                
                viz_files = self.list_visualization_files()
                if viz_files:
                    print(f"\nServing {len(viz_files)} visualizations:")
                    for viz_file in viz_files[:3]:
                        print(f"  http://192.168.3.127:8000/{viz_file}")
                    if len(viz_files) > 3:
                        print(f"  ... and {len(viz_files) - 3} more")
                else:
                    print("\nNo visualizations found.")
                    print("Create some using VisualizationManager option 8.")
            else:
                print("Failed to start server")
        
        except Exception as e:
            print(f"Error starting server: {e}")
        
        input("Press Enter to continue...")
    
    def stop_web_server(self):
        """Stop the web server"""
        print("\nStopping Web Visualization Server...")
        
        if self.check_web_server_status() == "Stopped":
            print("Server is not running")
            input("Press Enter to continue...")
            return
        
        try:
            import subprocess
            
            # Kill server processes
            subprocess.run(['pkill', '-f', 'python.*8000'], capture_output=True)
            subprocess.run(['pkill', '-f', 'http.server'], capture_output=True)
            
            # Give it time to stop
            import time
            time.sleep(1)
            
            if self.check_web_server_status() == "Stopped":
                print("✓ Web server stopped successfully")
            else:
                print("Server may still be running")
        
        except Exception as e:
            print(f"Error stopping server: {e}")
        
        input("Press Enter to continue...")
    
    def show_web_server_status(self):
        """Show detailed server status"""
        print("\nWeb Server Status")
        print("=" * 25)
        
        status = self.check_web_server_status()
        print(f"Status: {status}")
        
        if status == "Running":
            print("URL: http://192.168.3.127:8000/")
            print("Port: 8000")
            print("Directory: visualizations/")
            
            file_info = self.get_visualization_file_info()
            print(f"Visualization files: {file_info['count']}")
            print(f"Total size: {file_info['total_size']:.1f} KB")
            if file_info['oldest']:
                print(f"Oldest file: {file_info['oldest'].strftime('%Y-%m-%d %H:%M')}")
                print(f"Newest file: {file_info['newest'].strftime('%Y-%m-%d %H:%M')}")
        
        input("Press Enter to continue...")
    
    def list_visualizations(self):
        """List available visualization files"""
        print("\nAvailable Visualizations")
        print("=" * 30)
        
        viz_files = self.list_visualization_files()
        if not viz_files:
            print("No visualization files found")
            print("Create visualizations using:")
            print("  Main Menu → 8. Data Visualization & Monitoring")
        else:
            for i, viz_file in enumerate(viz_files, 1):
                print(f"{i:2d}. {viz_file}")
                print(f"     http://192.168.3.127:8000/{viz_file}")
        
        input("Press Enter to continue...")
    

    def delete_old_visualizations(self):
        """Delete visualizations older than specified days"""
        print("\nDelete Old Visualizations")
        print("=" * 30)
        
        viz_files = self.list_visualization_files()
        if not viz_files:
            print("No visualization files found")
            input("Press Enter to continue...")
            return
        
        print(f"Found {len(viz_files)} visualization files")
        
        try:
            days = input("Delete files older than how many days? (default 7): ").strip()
            if not days:
                days = 7
            else:
                days = int(days)
        except ValueError:
            print("Invalid number")
            input("Press Enter to continue...")
            return
        
        import time
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        old_files = []
        
        viz_dir = "visualizations"
        for viz_file in viz_files:
            file_path = os.path.join(viz_dir, viz_file)
            if os.path.getmtime(file_path) < cutoff_time:
                old_files.append(viz_file)
        
        if not old_files:
            print(f"No files older than {days} days found")
            input("Press Enter to continue...")
            return
        
        print(f"\nFiles to delete ({len(old_files)} files older than {days} days):")
        for old_file in old_files:
            file_path = os.path.join(viz_dir, old_file)
            mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            print(f"  {old_file} ({mtime.strftime('%Y-%m-%d %H:%M')})")
        
        if input("\nDelete these files? (y/N): ").lower() == 'y':
            deleted_count = 0
            for old_file in old_files:
                try:
                    file_path = os.path.join(viz_dir, old_file)
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {old_file}: {e}")
            
            print(f"✓ Deleted {deleted_count} old visualization files")
        else:
            print("Deletion cancelled")
        
        input("Press Enter to continue...")
    
    def delete_all_visualizations(self):
        """Delete all visualization files"""
        print("\nDelete All Visualizations")
        print("=" * 30)
        
        viz_files = self.list_visualization_files()
        if not viz_files:
            print("No visualization files found")
            input("Press Enter to continue...")
            return
        
        print(f"Found {len(viz_files)} visualization files:")
        viz_dir = "visualizations"
        total_size = 0
        
        for viz_file in viz_files:
            file_path = os.path.join(viz_dir, viz_file)
            size = os.path.getsize(file_path) / 1024
            total_size += size
            mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            print(f"  {viz_file} ({size:.1f} KB, {mtime.strftime('%Y-%m-%d %H:%M')})")
        
        print(f"\nTotal size: {total_size:.1f} KB")
        
        if input("\nDELETE ALL visualization files? (y/N): ").lower() == 'y':
            deleted_count = 0
            for viz_file in viz_files:
                try:
                    file_path = os.path.join(viz_dir, viz_file)
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {viz_file}: {e}")
            
            print(f"✓ Deleted {deleted_count} visualization files")
            print(f"✓ Freed {total_size:.1f} KB of disk space")
        else:
            print("Deletion cancelled")
        
        input("Press Enter to continue...")
    
    def get_visualization_file_info(self):
        """Get detailed info about visualization files"""
        viz_files = self.list_visualization_files()
        if not viz_files:
            return {"count": 0, "total_size": 0, "oldest": None, "newest": None}
        
        viz_dir = "visualizations"
        total_size = 0
        oldest_time = float('inf')
        newest_time = 0
        
        for viz_file in viz_files:
            file_path = os.path.join(viz_dir, viz_file)
            size = os.path.getsize(file_path)
            mtime = os.path.getmtime(file_path)
            
            total_size += size
            if mtime < oldest_time:
                oldest_time = mtime
            if mtime > newest_time:
                newest_time = mtime
        
        return {
            "count": len(viz_files),
            "total_size": total_size / 1024,  # KB
            "oldest": datetime.fromtimestamp(oldest_time) if oldest_time != float('inf') else None,
            "newest": datetime.fromtimestamp(newest_time) if newest_time > 0 else None
        }


    def list_visualization_files(self):
        """Get list of HTML files in visualizations directory"""
        try:
            viz_dir = "visualizations"
            if os.path.exists(viz_dir):
                files = [f for f in os.listdir(viz_dir) if f.endswith('.html')]
                return sorted(files, key=lambda x: os.path.getmtime(os.path.join(viz_dir, x)), reverse=True)
            return []
        except:
            return []


    def shutdown(self):
        """Graceful system shutdown"""
        print("\nShutting down system...")

        # Check for active jobs
        if hasattr(self.core, 'active_jobs') and self.core.active_jobs:
            print(f"WARNING: {len(self.core.active_jobs)} active jobs detected")
            if not self.core.confirm_operation("Shutdown with active jobs"):
                return

        # Shutdown all modules
        for module_name, module in self.modules.items():
            try:
                if hasattr(module, 'shutdown'):
                    module.shutdown()
                    print(f"  {module_name}: shutdown complete")
            except Exception as e:
                print(f"  {module_name}: shutdown error - {e}")

        # Core system cleanup
        self.core.shutdown()
        print("System shutdown complete")


def main():
    """Main entry point with error handling"""
    try:
        system = UnifiedSystem()
        system.main_menu()
    except KeyboardInterrupt:
        print("\n\nSystem interrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Goodbye!")


def analyze_lottery_data_with_mt():
    """Analyze lottery data specifically for MT19937 patterns"""
    print("\n" + "="*60)
    print("LOTTERY DATA MT19937 ANALYSIS")
    print("="*60)

    # Load your lottery data
    try:
        import json
        with open('daily3.json', 'r') as f:
            lottery_data = json.load(f)

        # Extract draw numbers
        draws = [entry.get('draw') for entry in lottery_data if entry.get('draw') is not None]
        print(f"Loaded {len(draws)} lottery draws from daily3.json")
        print(f"Sample draws: {draws[:10]}")

        if len(draws) < 50:
            print("Need at least 50 draws for meaningful MT19937 analysis")
            return

        # Import the working MT reconstruction
        from advanced_mt_reconstruction import create_advanced_mt_reconstruction_analysis

        # Analyze with MT19937 reconstruction
        print(f"\nAnalyzing {len(draws)} lottery draws for MT19937 patterns...")
        result = create_advanced_mt_reconstruction_analysis(draws)

        print(f"\nMT19937 ANALYSIS RESULTS:")
        print(f"Success: {result['success']}")
        print(f"Method: {result.get('method', 'Unknown')}")
        print(f"Confidence: {result.get('confidence', 0)}")

        if 'mt_property_tests' in result:
            tests = result['mt_property_tests']
            print(f"Likely MT19937: {tests.get('likely_mt19937', False)}")
            print(f"Bit balance: {tests.get('bit_balance', 'N/A')}")
            print(f"Uniformity test: {tests.get('uniformity_test', 'N/A')}")

        if 'verification' in result and result['verification']:
            verification = result['verification']
            print(f"Verification: {verification.get('verified', False)}")
            print(f"Match rate: {verification.get('match_rate', 0)*100:.1f}%")

        # Also run through gap-aware analysis
        print(f"\n" + "="*40)
        print("RUNNING COMPLETE GAP-AWARE ANALYSIS")
        print("="*40)

        from gap_aware_prng_reconstruction import create_comprehensive_gap_analysis
        gap_result = create_comprehensive_gap_analysis(draws, algorithm_types=['mt'])

        print("Gap-Aware Analysis Results:")
        if gap_result.get('success', False):
            mt_analysis = gap_result.get('analyses', {}).get('mt', {})
            print(f"MT Analysis Success: {mt_analysis.get('success', False)}")
            print(f"MT Confidence: {mt_analysis.get('confidence', 0)}")

        return result

    except FileNotFoundError:
        print("daily3.json not found. Make sure your lottery data file exists.")
    except Exception as e:
        print(f"Error analyzing lottery data: {e}")
        import traceback
        traceback.print_exc()


def enhanced_main_menu():
    """Enhanced main menu with lottery analysis"""
    while True:
        print("\n" + "="*50)
        print("DISTRIBUTED PRNG ANALYSIS SYSTEM")
        print("="*50)
        print("1. Test Systems")
        print("2. Advanced Research Jobs")
        print("3. Gap-Aware Analysis")
        print("4. Analyze Your Lottery Data (MT19937)")  # Preserved functionality
        print("5. Exit")

        choice = input("\nSelect option (1-5): ").strip()

        if choice == '1':
            try:
                from test_systems import test_all_systems
                test_all_systems()
            except ImportError:
                print("test_systems module not available")
        elif choice == '2':
            try:
                from advanced_research import advanced_research_menu
                advanced_research_menu()
            except ImportError:
                print("advanced_research module not available")
        elif choice == '3':
            try:
                from gap_aware_analysis import gap_aware_analysis_menu
                gap_aware_analysis_menu()
            except ImportError:
                print("gap_aware_analysis module not available")
        elif choice == '4':
            analyze_lottery_data_with_mt()  # Preserved function
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1-5.")


def lottery_analysis_menu():
    """Lottery-specific analysis menu - preserved functionality"""
    while True:
        print("\n" + "="*50)
        print("LOTTERY DATA ANALYSIS")
        print("="*50)
        print("1. MT19937 Pattern Analysis")
        print("2. Quick Statistical Check")
        print("3. All Conversion Methods Test")
        print("4. Back to Main Menu")

        choice = input("\nSelect option (1-4): ").strip()

        if choice == '1':
            # Run MT19937 analysis with method selection
            try:
                import analyze_my_lottery_data
                analyze_my_lottery_data.main()
            except ImportError:
                print("Error: analyze_my_lottery_data.py not found")
            except Exception as e:
                print(f"Analysis failed: {e}")

        elif choice == '2':
            # Quick statistical check
            try:
                import json
                with open('daily3.json', 'r') as f:
                    data = json.load(f)
                draws = [entry.get('draw') for entry in data if entry.get('draw') is not None]

                print(f"\nQuick Statistics:")
                print(f"Total draws: {len(draws)}")
                print(f"Range: {min(draws)} to {max(draws)}")
                print(f"Average: {sum(draws)/len(draws):.2f}")
                print(f"Unique values: {len(set(draws))}")

                # Quick MT test with hash conversion
                from analyze_my_lottery_data import convert_lottery_to_32bit
                from advanced_mt_reconstruction import AdvancedMT19937Reconstructor

                converted = convert_lottery_to_32bit(draws, 'hash')
                reconstructor = AdvancedMT19937Reconstructor()
                result = reconstructor.reconstruct_mt_state(converted[:min(200, len(converted))])

                print(f"\nQuick MT19937 Check:")
                print(f"Analysis: {result.get('method', 'Unknown')}")
                print(f"Likely MT19937: {result.get('likely_mt', 'Unknown')}")

            except Exception as e:
                print(f"Quick check failed: {e}")

        elif choice == '3':
            # Test all conversion methods
            try:
                import analyze_my_lottery_data
                import json

                with open('daily3.json', 'r') as f:
                    data = json.load(f)
                draws = [entry.get('draw') for entry in data if entry.get('draw') is not None]

                methods = ['hash', 'multiply', 'shift_combine', 'sequence']
                print(f"\nTesting all conversion methods with {len(draws)} draws:")

                for method in methods:
                    converted = analyze_my_lottery_data.convert_lottery_to_32bit(draws, method)
                    from advanced_mt_reconstruction import AdvancedMT19937Reconstructor
                    reconstructor = AdvancedMT19937Reconstructor()
                    result = reconstructor.reconstruct_mt_state(converted[:min(200, len(converted))])

                    likely_mt = result.get('likely_mt', False)
                    confidence = result.get('confidence', 0) * 100
                    print(f"  {method:15} | MT19937-like: {str(likely_mt):5} | Confidence: {confidence:5.1f}%")

            except Exception as e:
                print(f"All methods test failed: {e}")

        elif choice == '4':
            break
        else:
            print("Invalid choice. Please select 1-4.")


if __name__ == "__main__":
    main()
