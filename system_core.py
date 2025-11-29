#!/usr/bin/env python3
"""
System Core Module - Central system functionality and shared utilities

This module provides the core functionality that all other modules depend on,
including configuration, logging, file verification, and common utilities.
"""

import os
import sys
import json
import subprocess
import time
import glob
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Import database components
try:
    from database_system import get_database
    from advanced_search_manager import AdvancedSearchManager
except ImportError as e:
    print(f"Warning: Could not import database components: {e}")
    print("Some features may not be available")

class SystemCore:
    """
    Core system functionality shared across all modules
    """

    def __init__(self):
        """Initialize core system components"""
        # Version and metadata
        self.version = "4.0.0 (Modular)"
        self.last_modified = "September 10, 2025"
        self.startup_time = datetime.now()

        # Core system files
        self.coordinator_script = "coordinator.py"
        self.worker_script = "distributed_worker.py"
        self.engine_script = "enhanced_gpu_model_id.py"
        self.config_file = "distributed_config.json"
        self.data_file = "daily3.json"
        self.db_path = "prng_analysis.db"

        # System state
        self.active_jobs = {}
        self.performance_history = []
        self.error_log = []
        self.operation_count = 0

        # Configuration
        self.max_concurrent_jobs = 3
        self.default_timeout = 3600
        self.monitoring_interval = 5
        self.backup_retention_days = 30
        self.log_retention_days = 7

        # Cluster specifications
        self.cluster_specs = {
            "total_gpus": 26,
            "zeus_gpus": 2,
            "rig6600_gpus": 12,
            "rig6600b_gpus": 12,
            "total_tflops": 285.69,
            "architecture": "Mixed CUDA/ROCm"
        }

        # Initialize system
        self._initialize_core()

    def _initialize_core(self):
        """Initialize core system components"""
        try:
            print("Initializing core system...")

            # Verify system files
            self.verify_system_files()

            # Setup directories
            self.setup_directories()

            # Initialize logging
            self.setup_logging()

            # Verify permissions
            self.verify_permissions()

            # Initialize database connection
            try:
                self.db = get_database(self.db_path)
                self.search_manager = AdvancedSearchManager(self.db_path)
                print("Database components initialized")
            except Exception as e:
                print(f"Warning: Database initialization failed: {e}")
                self.db = None
                self.search_manager = None

            # Initialize modules
            self._initialize_modules()

            self.log_operation("core_init", "SUCCESS", "Core system initialized")
            print("Core system initialization complete")

        except Exception as e:
            self.log_operation("core_init", "ERROR", f"Core initialization failed: {e}")
            print(f"CRITICAL: Core initialization failed: {e}")
            raise

    def _initialize_modules(self):
        """Initialize all system modules"""
        try:
            # Import and initialize modules
            from modules.direct_analysis import DirectAnalysis
            from modules.database_manager import DatabaseManager
            from modules.result_viewer import ResultViewer
            from modules.system_monitor import SystemMonitor
            from modules.advanced_research import AdvancedResearch
            from modules.file_manager import FileManager
            from modules.performance_analytics import PerformanceAnalytics
            from modules.visualization_manager import VisualizationManager
            
            # Initialize module instances
            self.direct_analysis = DirectAnalysis(self)
            self.database_manager = DatabaseManager(self)
            self.result_viewer = ResultViewer(self)
            self.system_monitor = SystemMonitor(self)
            self.advanced_research = AdvancedResearch(self)
            self.file_manager = FileManager(self)
            self.performance_analytics = PerformanceAnalytics(self)
            self.visualization = VisualizationManager(self)
            
            print("All modules initialized successfully")
            
        except ImportError as e:
            print(f"Warning: Some modules could not be loaded: {e}")
            print("Some functionality may not be available")
        except Exception as e:
            print(f"Error initializing modules: {e}")

    def verify_system_files(self):
        """Verify required system files exist and are valid"""
        print("Verifying system files...")

        required_files = {
            self.coordinator_script: {"type": "python", "required": True, "min_size": 1000},
            self.worker_script: {"type": "python", "required": True, "min_size": 1000},
            self.engine_script: {"type": "python", "required": True, "min_size": 1000},
            self.config_file: {"type": "json", "required": True, "min_size": 100},
            self.data_file: {"type": "json", "required": True, "min_size": 100},
            "database_system.py": {"type": "python", "required": False, "min_size": 1000},
            "advanced_search_manager.py": {"type": "python", "required": False, "min_size": 1000},
            "process_db_jobs.py": {"type": "python", "required": False, "min_size": 1000}
        }

        missing_required = []
        for filename, specs in required_files.items():
            if not os.path.exists(filename):
                if specs["required"]:
                    missing_required.append(filename)
                    print(f"  {filename}: MISSING (REQUIRED)")
                else:
                    print(f"  {filename}: MISSING (OPTIONAL)")
            else:
                try:
                    file_size = os.path.getsize(filename)
                    if file_size < specs["min_size"]:
                        print(f"  {filename}: TOO SMALL ({file_size} bytes)")
                        continue

                    # Basic validation
                    if specs["type"] == "json":
                        with open(filename, 'r') as f:
                            json.load(f)
                    elif specs["type"] == "python":
                        subprocess.run(['python3', '-m', 'py_compile', filename],
                                     check=True, capture_output=True)

                    print(f"  {filename}: OK ({file_size} bytes)")

                except Exception as e:
                    print(f"  {filename}: ERROR - {e}")

        if missing_required:
            raise FileNotFoundError(f"Missing required files: {missing_required}")

    def setup_directories(self):
        """Create necessary directory structure"""
        directories = [
            "modules",      # Module files
            "logs",         # System logs
            "backups",      # File backups
            "archives",     # Result archives
            "temp",         # Temporary files
            "exports",      # Exported results
            "jobs",         # Job files
            "results",      # Analysis results
            "reports"       # Generated reports
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def setup_logging(self):
        """Initialize logging system"""
        self.log_file = f"logs/system_{datetime.now().strftime('%Y%m%d')}.log"

        try:
            with open(self.log_file, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"SYSTEM STARTUP: {datetime.now().isoformat()}\n")
                f.write(f"Version: {self.version}\n")
                f.write(f"{'='*60}\n")
        except Exception as e:
            print(f"Warning: Logging setup failed: {e}")
            self.log_file = None

    def verify_permissions(self):
        """Verify system has necessary permissions"""
        # Test write permissions
        try:
            test_file = "temp_permission_test.txt"
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            raise PermissionError(f"Write permission test failed: {e}")

        # Test Python execution
        try:
            result = subprocess.run(['python3', '--version'], capture_output=True)
            if result.returncode != 0:
                raise RuntimeError("Python execution test failed")
        except Exception as e:
            raise RuntimeError(f"Python execution test failed: {e}")

    def log_operation(self, operation_type: str, status: str, details: str):
        """Log system operations"""
        timestamp = datetime.now().isoformat()
        log_entry = f"{timestamp} | {operation_type} | {status} | {details}"

        # Add to memory log
        self.error_log.append({
            'timestamp': timestamp,
            'operation': operation_type,
            'status': status,
            'details': details
        })

        # Write to file
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(log_entry + "\n")
            except:
                pass  # Silent fail for logging

        # Keep memory log manageable
        if len(self.error_log) > 1000:
            self.error_log = self.error_log[-500:]

    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self):
        """Print system header with status"""
        print("=" * 70)
        print("DISTRIBUTED PRNG ANALYSIS SYSTEM (MODULAR)")
        print(f"Version: {self.version} | Modified: {self.last_modified}")
        print(f"Cluster: {self.cluster_specs['total_gpus']} GPUs | {self.cluster_specs['total_tflops']} TFLOPS")
        print(f"Architecture: {self.cluster_specs['architecture']}")

        # System status
        uptime = datetime.now() - self.startup_time
        print(f"Uptime: {uptime} | Operations: {self.operation_count}")

        # Active jobs
        if self.active_jobs:
            print(f"Active Jobs: {len(self.active_jobs)}")

        # Database status
        if self.db:
            try:
                stats = self.db.get_statistics()
                print(f"DB: {stats['total_cached_results']:,} cached results | {stats['database_size_mb']:.1f} MB")
            except:
                print("DB: Connected")
        else:
            print("DB: Not available")

        print("=" * 70)

    def display_system_status(self):
        """Display brief system status"""
        try:
            # Check active processes
            ps_result = subprocess.run(['pgrep', '-f', 'coordinator.py'],
                                     capture_output=True, text=True)
            active_processes = len(ps_result.stdout.strip().split('\n')) if ps_result.stdout.strip() else 0

            # Check recent files
            recent_results = len([f for f in os.listdir('.') if 'results' in f and
                                (time.time() - os.path.getmtime(f)) < 3600])

            # DB pending jobs
            pending_jobs = 0
            if self.db:
                try:
                    stats = self.db.get_statistics()
                    pending_jobs = stats['jobs_by_status'].get('pending', 0)
                except:
                    pass

            # Display summary
            if active_processes > 0 or recent_results > 0 or pending_jobs > 0:
                print(f"\nStatus: {active_processes} active processes, {recent_results} recent results, {pending_jobs} pending jobs")
        except:
            pass

    def confirm_operation(self, operation_description: str, double_confirm: bool = False) -> bool:
        """Get user confirmation for operations"""
        confirm = input(f"\nConfirm: {operation_description}? (y/N): ")
        if confirm.lower() != 'y':
            return False

        if double_confirm:
            final_confirm = input("Are you absolutely sure? This cannot be undone (yes/no): ")
            return final_confirm.lower() == 'yes'

        return True

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            'version': self.version,
            'uptime': str(datetime.now() - self.startup_time),
            'operation_count': self.operation_count,
            'active_jobs': len(self.active_jobs),
            'cluster_specs': self.cluster_specs,
            'db_available': self.db is not None,
            'search_manager_available': self.search_manager is not None,
            'log_file': self.log_file,
            'error_count': len([e for e in self.error_log if e['status'] == 'ERROR'])
        }

    def cleanup_temp_files(self):
        """Clean up temporary files"""
        temp_patterns = ["temp_*.json", "*.tmp", "*.temp"]
        for pattern in temp_patterns:
            for file in glob.glob(pattern):
                try:
                    os.remove(file)
                except:
                    pass

    def shutdown(self):
        """Core system shutdown"""
        print("Shutting down core system...")

        # Cleanup temp files
        self.cleanup_temp_files()

        # Log shutdown
        self.log_operation("core_shutdown", "SUCCESS", "Core system shutdown")

        print("Core system shutdown complete")
