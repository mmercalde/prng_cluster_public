#!/usr/bin/env python3
"""
Enhanced Comprehensive Backup Script to SER8 Remote Server
Uses AST parsing to find ALL Python dependencies recursively
Target: michael@192.168.3.24:~/cluster_shared
"""

import os
import sys
import subprocess
import json
import ast
from pathlib import Path
from datetime import datetime
from typing import Set, List, Dict

class DependencyFinder:
    """Find all Python file dependencies using AST parsing"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.discovered = set()
        self.pending = set()
        
    def find_imports(self, filepath: str) -> Set[str]:
        """Extract all local imports from a Python file using AST"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                tree = ast.parse(f.read(), filename=filepath)
            
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
            
            return imports
        except Exception as e:
            print(f"  Warning: Could not parse {filepath}: {e}")
            return set()
    
    def resolve_local_file(self, module_name: str) -> List[str]:
        """Try to find local Python files for a module name"""
        candidates = [
            f"{module_name}.py",
            f"{module_name}/__init__.py",
            f"modules/{module_name}.py",
            f"modules/{module_name}/__init__.py",
        ]
        
        found = []
        for candidate in candidates:
            full_path = self.base_dir / candidate
            if full_path.exists():
                found.append(str(full_path))
        
        return found
    
    def discover_recursive(self, start_files: List[str]) -> Set[str]:
        """Recursively discover all dependencies"""
        self.pending.update(start_files)
        
        while self.pending:
            current_file = self.pending.pop()
            
            if current_file in self.discovered:
                continue
                
            self.discovered.add(current_file)
            
            # Find imports in this file
            imports = self.find_imports(current_file)
            
            # Resolve imports to local files
            for module_name in imports:
                local_files = self.resolve_local_file(module_name)
                for local_file in local_files:
                    if local_file not in self.discovered:
                        self.pending.add(local_file)
        
        return self.discovered


class BackupManager:
    def __init__(self):
        self.remote_user = "michael"
        self.remote_host = "192.168.3.24"
        self.remote_password = "password"
        self.remote_path = "~/cluster_shared/prng_backup"
        
        # Core entry point files - START HERE
        self.entry_points = [
            'coordinator.py',
            'unified_system_working.py',
            'complete_whitepaper_workflow_with_meta_optimizer.py',
            'window_optimizer.py',
            'window_optimizer_bayesian.py',
            'window_optimizer_integration_final.py',
            'run_scorer_meta_optimizer.sh',
            'generate_scorer_jobs.py',
            'scorer_trial_worker.py',
            'reinforcement_engine.py',
            'sieve_filter.py',
            'reverse_sieve_filter.py',
            'distributed_worker.py',
            'synthetic_pattern_generator.py',
            'synthetic_pattern_generator_enhanced.py',
            'adaptive_meta_optimizer.py',
            'meta_prediction_optimizer_anti_overfit.py',
            'generate_full_scoring_jobs.py',
            'survivor_scorer.py',
            'anti_overfit_trial_worker.py',
            'analyze_my_lottery_data.py',
            'advanced_mt_reconstruction.py',
            'gap_aware_prng_reconstruction.py',
            'test_systems.py',
            'advanced_research.py',
            'gap_aware_analysis.py',
        ]
        
        # Data files
        self.data_files = [
            'synthetic_lottery.json',
            'daily3.json',
            'bidirectional_survivors.json',
            'forward_survivors.json',
            'reverse_survivors.json',
            'optimal_window_config.json',
            'optimal_scorer_config.json',
            'distributed_config.json',
            'ml_coordinator_config.json',
            'reinforcement_engine_config.json',
            'train_history.json',
            'holdout_history.json',
        ]
        
        # Shell scripts
        self.shell_scripts = [
            'run_scorer_meta_optimizer.sh',
            'run_full_scoring.sh',
            'run_ml_distributed.sh',
        ]
        
        # Modules directory - backup entire directory
        self.modules_dir = 'modules'
        
        # Results directories (limited to recent files)
        self.result_dirs = ['results', 'optuna_studies', '.recovery']
        
    def check_prerequisites(self):
        """Check if sshpass is available"""
        print("Checking prerequisites...")
        try:
            subprocess.run(['which', 'sshpass'], 
                         check=True, 
                         capture_output=True)
            print("✅ sshpass available")
            return True
        except subprocess.CalledProcessError:
            print("❌ sshpass not found")
            print("Install with: sudo apt-get install sshpass")
            return False
    
    def discover_all_dependencies(self):
        """Use AST parsing to find ALL Python dependencies"""
        print("\n🔍 Discovering all Python dependencies recursively...")
        
        finder = DependencyFinder()
        
        # Start with entry points
        existing_entry_points = [f for f in self.entry_points if os.path.exists(f)]
        print(f"  Starting from {len(existing_entry_points)} entry points...")
        
        # Recursively discover
        all_python_files = finder.discover_recursive(existing_entry_points)
        
        print(f"  ✅ Discovered {len(all_python_files)} Python files total")
        
        return sorted(all_python_files)
    
    def discover_modules_directory(self):
        """Discover all files in modules/ directory"""
        print("\n📁 Scanning modules directory...")
        modules_files = []
        
        if os.path.exists(self.modules_dir):
            for root, dirs, files in os.walk(self.modules_dir):
                for file in files:
                    if file.endswith(('.py', '.json', '.txt', '.md')):
                        full_path = os.path.join(root, file)
                        modules_files.append(full_path)
            
            print(f"  ✅ Found {len(modules_files)} files in modules/")
        else:
            print(f"  ⚠️  modules/ directory not found")
        
        return modules_files
    
    def create_backup_manifest(self):
        """Create manifest of all files to backup"""
        print("\n📋 Creating backup manifest...")
        
        manifest = {
            'timestamp': datetime.now().isoformat(),
            'source_host': os.uname().nodename,
            'backup_method': 'recursive_ast_discovery',
            'files': {
                'python_files': [],
                'shell_scripts': [],
                'data_files': [],
                'modules': [],
                'results': []
            },
            'statistics': {}
        }
        
        # 1. Discover ALL Python dependencies
        python_files = self.discover_all_dependencies()
        manifest['files']['python_files'] = python_files
        print(f"  ✅ Python files: {len(python_files)}")
        
        # 2. Shell scripts
        for script in self.shell_scripts:
            if os.path.exists(script):
                manifest['files']['shell_scripts'].append(script)
                print(f"  ✅ Shell: {script}")
        
        # 3. Data files
        for data_file in self.data_files:
            if os.path.exists(data_file):
                manifest['files']['data_files'].append(data_file)
                print(f"  ✅ Data: {data_file}")
        
        # 4. Modules directory
        modules = self.discover_modules_directory()
        manifest['files']['modules'] = modules
        
        # 5. Recent results (last 100 files from each dir)
        print("\n📊 Collecting recent results...")
        for result_dir in self.result_dirs:
            if os.path.exists(result_dir):
                try:
                    files = []
                    for root, dirs, filenames in os.walk(result_dir):
                        for filename in filenames:
                            full_path = os.path.join(root, filename)
                            files.append(full_path)
                    
                    # Sort by modification time, take most recent 100
                    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    recent_files = files[:100]
                    manifest['files']['results'].extend(recent_files)
                    print(f"  ✅ {result_dir}: {len(recent_files)} recent files")
                except Exception as e:
                    print(f"  ⚠️  Error scanning {result_dir}: {e}")
        
        # Calculate statistics
        total_files = sum(len(v) for v in manifest['files'].values() if isinstance(v, list))
        manifest['statistics'] = {
            'total_files': total_files,
            'python_files': len(manifest['files']['python_files']),
            'shell_scripts': len(manifest['files']['shell_scripts']),
            'data_files': len(manifest['files']['data_files']),
            'modules': len(manifest['files']['modules']),
            'results': len(manifest['files']['results'])
        }
        
        # Save manifest
        manifest_file = 'backup_manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n📋 Manifest saved: {manifest_file}")
        print(f"📊 Total files to backup: {total_files}")
        
        return manifest
    
    def create_backup_archive(self, manifest):
        """Create tar.gz archive of all files"""
        print("\n📦 Creating backup archive...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_name = f'prng_backup_{timestamp}.tar.gz'
        
        # Collect all files
        all_files = []
        for category, files in manifest['files'].items():
            if isinstance(files, list):
                all_files.extend(files)
        
        # Add manifest itself
        all_files.append('backup_manifest.json')
        
        # Remove duplicates and non-existent files
        all_files = list(set(all_files))
        all_files = [f for f in all_files if os.path.exists(f)]
        
        print(f"  Creating archive with {len(all_files)} files...")
        
        try:
            # Create tar command with file list
            with open('.backup_filelist.txt', 'w') as f:
                for file in all_files:
                    f.write(f"{file}\n")
            
            cmd = ['tar', '-czf', archive_name, '-T', '.backup_filelist.txt']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Clean up file list
            os.remove('.backup_filelist.txt')
            
            if result.returncode == 0:
                size_mb = os.path.getsize(archive_name) / (1024 * 1024)
                print(f"  ✅ Archive created: {archive_name} ({size_mb:.2f} MB)")
                return archive_name
            else:
                print(f"  ❌ Error creating archive: {result.stderr}")
                return None
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return None
    
    def upload_to_remote(self, archive_name):
        """Upload archive to remote server using sshpass + scp"""
        print(f"\n📤 Uploading {archive_name} to {self.remote_host}...")
        
        # Create remote directory
        print("  Creating remote directory...")
        mkdir_cmd = [
            'sshpass', '-p', self.remote_password,
            'ssh', '-o', 'StrictHostKeyChecking=no',
            f'{self.remote_user}@{self.remote_host}',
            f'mkdir -p {self.remote_path}'
        ]
        
        subprocess.run(mkdir_cmd, capture_output=True)
        
        # Upload archive
        print("  Uploading archive...")
        scp_cmd = [
            'sshpass', '-p', self.remote_password,
            'scp', '-o', 'StrictHostKeyChecking=no',
            archive_name,
            f'{self.remote_user}@{self.remote_host}:{self.remote_path}/'
        ]
        
        result = subprocess.run(scp_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  ✅ Upload complete!")
            print(f"  📍 Location: {self.remote_user}@{self.remote_host}:{self.remote_path}/{archive_name}")
            return True
        else:
            print(f"  ❌ Upload failed: {result.stderr}")
            return False
    
    def extract_on_remote(self, archive_name):
        """Extract archive on remote server"""
        print(f"\n📂 Extracting archive on remote server...")
        
        extract_cmd = [
            'sshpass', '-p', self.remote_password,
            'ssh', '-o', 'StrictHostKeyChecking=no',
            f'{self.remote_user}@{self.remote_host}',
            f'cd {self.remote_path} && tar -xzf {archive_name}'
        ]
        
        result = subprocess.run(extract_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  ✅ Extraction complete!")
            print(f"  📂 Files extracted to: {self.remote_path}/")
            return True
        else:
            print(f"  ❌ Extraction failed: {result.stderr}")
            return False
    
    def verify_backup(self, archive_name):
        """Verify backup on remote server"""
        print(f"\n✓ Verifying backup on remote server...")
        
        verify_cmd = [
            'sshpass', '-p', self.remote_password,
            'ssh', '-o', 'StrictHostKeyChecking=no',
            f'{self.remote_user}@{self.remote_host}',
            f'ls -lh {self.remote_path}/{archive_name}'
        ]
        
        result = subprocess.run(verify_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  ✅ Backup verified:")
            print(f"  {result.stdout}")
            return True
        else:
            print(f"  ❌ Verification failed: {result.stderr}")
            return False
    
    def cleanup_local(self, archive_name, keep_archive=False):
        """Clean up local files"""
        print("\n🧹 Cleaning up local files...")
        
        if not keep_archive and os.path.exists(archive_name):
            try:
                os.remove(archive_name)
                print(f"  ✅ Removed local archive: {archive_name}")
            except Exception as e:
                print(f"  ⚠️  Could not remove archive: {e}")
        else:
            print(f"  📦 Keeping local archive: {archive_name}")
        
        if os.path.exists('backup_manifest.json'):
            print(f"  📋 Keeping manifest: backup_manifest.json")
    
    def run_backup(self):
        """Main backup execution"""
        print("="*70)
        print("PRNG ANALYSIS SYSTEM - COMPREHENSIVE BACKUP TO SER8")
        print("="*70)
        print(f"Target: {self.remote_user}@{self.remote_host}:{self.remote_path}")
        print("Method: Recursive AST-based dependency discovery")
        print("="*70)
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Create manifest with recursive discovery
        manifest = self.create_backup_manifest()
        
        # Show summary
        print("\n" + "="*70)
        print("BACKUP SUMMARY")
        print("="*70)
        stats = manifest['statistics']
        print(f"  Python files:  {stats['python_files']:>6,}")
        print(f"  Shell scripts: {stats['shell_scripts']:>6,}")
        print(f"  Data files:    {stats['data_files']:>6,}")
        print(f"  Module files:  {stats['modules']:>6,}")
        print(f"  Result files:  {stats['results']:>6,}")
        print(f"  {'─'*30}")
        print(f"  TOTAL FILES:   {stats['total_files']:>6,}")
        print("="*70)
        
        # Confirm
        response = input("\nProceed with backup? (yes/no): ").strip().lower()
        if response != 'yes':
            print("Backup cancelled.")
            return False
        
        # Create archive
        archive_name = self.create_backup_archive(manifest)
        if not archive_name:
            return False
        
        # Upload to remote
        if not self.upload_to_remote(archive_name):
            return False
        
        # Extract on remote
        if not self.extract_on_remote(archive_name):
            return False
        
        # Verify
        if not self.verify_backup(archive_name):
            return False
        
        # Cleanup
        self.cleanup_local(archive_name, keep_archive=True)
        
        print("\n" + "="*70)
        print("✅✅✅ BACKUP COMPLETE! ✅✅✅")
        print("="*70)
        print(f"Archive: {archive_name}")
        print(f"Location: {self.remote_user}@{self.remote_host}:{self.remote_path}/")
        print(f"Manifest: backup_manifest.json")
        print(f"\nFiles backed up: {stats['total_files']:,}")
        print(f"Archive size: {os.path.getsize(archive_name)/(1024*1024):.2f} MB")
        print("\nTo restore:")
        print(f"  ssh {self.remote_user}@{self.remote_host}")
        print(f"  cd {self.remote_path}")
        print(f"  tar -xzf {archive_name}")
        print("="*70)
        
        return True


def main():
    try:
        backup = BackupManager()
        success = backup.run_backup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nBackup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
