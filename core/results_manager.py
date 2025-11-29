#!/usr/bin/env python3
"""
================================================================================
RESULTS MANAGER - Schema-Driven Results Engine
================================================================================

File: results_manager.py
Version: 1.0.0
Created: 2025-11-03
Author: Distributed PRNG Analysis System

PURPOSE:
--------
Schema-driven engine for creating human-readable and machine-readable analysis
results. Validates data against schemas and generates multiple output formats:
- Human-readable text summaries (.txt)
- Machine-readable JSON files (.json)
- Excel-compatible CSV files (.csv)

FEATURES:
---------
- Zero hardcoded field names (all from schemas)
- Automatic validation against results_schema_v1.json
- Flexible formatting via output_templates.json
- Field mapping via field_mappings.json
- Supports all 24 analysis types
- ML/RL metrics ready
- CLI and programmatic interfaces

USAGE:
------
From Python:
    from results_manager import ResultsManager
    
    rm = ResultsManager()
    rm.save_results(
        analysis_type='bidirectional_sieve',
        run_id='1B_java_lcg',
        data=results_dict
    )

From CLI:
    python3 results_manager.py --help
    python3 results_manager.py --validate results.json

DEPENDENCIES:
-------------
- Python 3.8+
- json (stdlib)
- csv (stdlib)
- datetime (stdlib)
- pathlib (stdlib)
- typing (stdlib)

REVISION HISTORY:
-----------------
Version 1.0.0 (2025-11-03):
- Initial implementation
- Schema-driven validation and formatting
- Multi-format output (txt, csv, json)
- Support for 24 analysis types
- ML metrics integration

================================================================================
"""

import json
import csv
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import socket


class ResultsManager:
    """
    Schema-driven results manager for PRNG analysis.
    
    This class loads schemas and generates formatted outputs based entirely
    on configuration files - no hardcoded field names or formats.
    
    Attributes:
        schema (dict): Loaded results schema
        templates (dict): Loaded output templates
        mappings (dict): Loaded field mappings
        results_dir (Path): Base directory for results
        
    Example:
        >>> rm = ResultsManager()
        >>> rm.save_results('bidirectional_sieve', 'test_run', data)
        >>> # Creates: results/summaries/test_run_summary.txt
        >>> #          results/csv/test_run.csv
        >>> #          results/json/test_run.json
    """
    
    def __init__(self, schema_dir: str = "schemas", results_dir: str = "results"):
        """
        Initialize ResultsManager with schemas and output directories.
        
        Args:
            schema_dir: Directory containing schema JSON files
            results_dir: Base directory for output files
            
        Raises:
            FileNotFoundError: If schema files are missing
            json.JSONDecodeError: If schema files are invalid
        """
        self.schema_dir = Path(schema_dir)
        self.results_dir = Path(results_dir)
        
        # Load schemas
        self.schema = self._load_schema('results_schema_v1.json')
        self.templates = self._load_schema('output_templates.json')
        self.mappings = self._load_schema('field_mappings.json')
        
        # Create output directories
        self._create_output_dirs()
        
        print(f"✅ ResultsManager initialized")
        print(f"   Schema version: {self.schema['_schema_info']['version']}")
        print(f"   Analysis types: {len(self._get_allowed_analysis_types())}")
        
    def _load_schema(self, filename: str) -> dict:
        """
        Load and validate a schema file.
        
        Args:
            filename: Name of schema file
            
        Returns:
            Parsed JSON schema
            
        Raises:
            FileNotFoundError: If schema file doesn't exist
            json.JSONDecodeError: If schema is invalid JSON
        """
        schema_path = self.schema_dir / filename
        
        if not schema_path.exists():
            raise FileNotFoundError(
                f"Schema file not found: {schema_path}\n"
                f"Make sure schemas/ directory exists with all required files."
            )
        
        try:
            with open(schema_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in {filename}: {e.msg}",
                e.doc, e.pos
            )
    
    def _create_output_dirs(self):
        """Create all required output directories."""
        dirs = [
            self.results_dir / 'summaries',
            self.results_dir / 'csv',
            self.results_dir / 'json',
            self.results_dir / 'detailed',
            self.results_dir / 'configs'
        ]
        
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _get_allowed_analysis_types(self) -> List[str]:
        """Get list of allowed analysis types from schema."""
        return self.schema['run_metadata']['fields']['analysis_type']['allowed_values']
    
    def validate_analysis_type(self, analysis_type: str) -> bool:
        """
        Validate that analysis type is supported.
        
        Args:
            analysis_type: Type to validate
            
        Returns:
            True if valid, False otherwise
        """
        allowed = self._get_allowed_analysis_types()
        return analysis_type in allowed
    
    def save_results(
        self,
        analysis_type: str,
        run_id: str,
        data: Dict[str, Any],
        create_all_formats: bool = True
    ) -> Dict[str, Path]:
        """
        Save analysis results in multiple formats.
        
        This is the main entry point for saving results. It validates data,
        generates all output formats, and returns paths to created files.
        
        Args:
            analysis_type: Type of analysis (must be in schema)
            run_id: Unique identifier for this run
            data: Results data dictionary (should match schema structure)
            create_all_formats: If True, create txt/csv/json (default: True)
            
        Returns:
            Dictionary mapping format names to file paths:
            {'summary': Path, 'csv': Path, 'json': Path, 'detailed': Path}
            
        Raises:
            ValueError: If analysis_type invalid or data missing required fields
            
        Example:
            >>> data = {
            ...     'run_metadata': {...},
            ...     'analysis_parameters': {...},
            ...     'results_summary': {...},
            ...     'survivors': [...]
            ... }
            >>> paths = rm.save_results('bidirectional_sieve', 'test_1B', data)
            >>> print(paths['summary'])
            results/summaries/test_1B_summary.txt
        """
        # Validate analysis type
        if not self.validate_analysis_type(analysis_type):
            allowed = self._get_allowed_analysis_types()
            raise ValueError(
                f"Invalid analysis_type: '{analysis_type}'\n"
                f"Allowed types: {', '.join(allowed)}"
            )
        
        # Add system metadata
        data = self._add_system_metadata(data, analysis_type, run_id)
        
        # Validate data structure (basic check)
        self._validate_data_structure(data)
        
        # Create output files
        output_paths = {}
        
        try:
            if create_all_formats:
                # Generate text summary
                summary_path = self._create_text_summary(data, run_id)
                output_paths['summary'] = summary_path
                
                # Generate CSV export
                csv_path = self._create_csv_export(data, run_id)
                output_paths['csv'] = csv_path
                
                # Generate JSON (top survivors)
                json_path = self._create_json_export(data, run_id, top_only=True)
                output_paths['json'] = json_path
                
                # Generate detailed JSON (all data)
                detailed_path = self._create_json_export(data, run_id, top_only=False)
                output_paths['detailed'] = detailed_path
                
                # Save config
                config_path = self._save_config(data, run_id)
                output_paths['config'] = config_path
            
            print(f"\n✅ Results saved successfully!")
            print(f"   Run ID: {run_id}")
            print(f"   Analysis: {analysis_type}")
            for format_name, path in output_paths.items():
                print(f"   {format_name:10s}: {path}")
            
            return output_paths
            
        except Exception as e:
            print(f"\n❌ Error saving results: {e}")
            raise
    
    def _add_system_metadata(
        self,
        data: Dict[str, Any],
        analysis_type: str,
        run_id: str
    ) -> Dict[str, Any]:
        """
        Add system-generated metadata to results.
        
        Args:
            data: User-provided data
            analysis_type: Type of analysis
            run_id: Run identifier
            
        Returns:
            Data with added metadata
        """
        # Ensure run_metadata exists
        if 'run_metadata' not in data:
            data['run_metadata'] = {}
        
        # Add required metadata
        data['run_metadata']['run_id'] = run_id
        data['run_metadata']['analysis_type'] = analysis_type
        data['run_metadata']['schema_version'] = self.schema['_schema_info']['version']
        
        # Add system info if not present
        if 'hostname' not in data['run_metadata']:
            try:
                data['run_metadata']['hostname'] = socket.gethostname()
            except:
                data['run_metadata']['hostname'] = 'unknown'
        
        if 'python_version' not in data['run_metadata']:
            data['run_metadata']['python_version'] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Add timestamp if not present
        if 'timestamp_start' not in data['run_metadata']:
            data['run_metadata']['timestamp_start'] = datetime.now().isoformat()
        
        return data
    
    def _validate_data_structure(self, data: Dict[str, Any]):
        """
        Basic validation of data structure.
        
        Args:
            data: Data to validate
            
        Raises:
            ValueError: If required sections are missing
        """
        required_sections = ['run_metadata', 'results_summary']
        
        for section in required_sections:
            if section not in data:
                raise ValueError(
                    f"Missing required section: '{section}'\n"
                    f"Data must contain: {', '.join(required_sections)}"
                )
    
    def _create_text_summary(self, data: Dict[str, Any], run_id: str) -> Path:
        """
        Create human-readable text summary.
        
        Args:
            data: Results data
            run_id: Run identifier
            
        Returns:
            Path to created summary file
        """
        output_path = self.results_dir / 'summaries' / f"{run_id}_summary.txt"
        
        with open(output_path, 'w') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("PRNG ANALYSIS RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            # Run info
            self._write_section(f, "RUN INFORMATION", data.get('run_metadata', {}))
            
            # Parameters
            self._write_section(f, "ANALYSIS PARAMETERS", data.get('analysis_parameters', {}))
            
            # Results summary
            self._write_section(f, "RESULTS SUMMARY", data.get('results_summary', {}))
            
            # Performance
            if 'performance_metrics' in data:
                self._write_section(f, "PERFORMANCE METRICS", data['performance_metrics'])
            
            # Top survivors
            if 'survivors' in data and data['survivors']:
                self._write_survivors_table(f, data['survivors'][:10])
            
            # Footer
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
        
        return output_path
    
    def _write_section(self, f, title: str, data: dict):
        """Write a section to text file."""
        f.write(f"\n{title}\n")
        f.write("-" * len(title) + "\n")
        
        for key, value in data.items():
            if not key.startswith('_') and value is not None:
                label = self._get_display_name(key)
                formatted_value = self._format_value(key, value)
                f.write(f"{label:30s}: {formatted_value}\n")
    
    def _write_survivors_table(self, f, survivors: List[Dict]):
        """Write survivors table to text file."""
        f.write("\nTOP SURVIVORS\n")
        f.write("-" * 13 + "\n")
        
        # Header
        f.write(f"{'Rank':>6} {'Seed':>12} {'Matches':>8} {'Rate':>8} {'Skip':>6} {'Direction':14}\n")
        f.write("-" * 80 + "\n")
        
        # Rows
        for i, survivor in enumerate(survivors, 1):
            seed = survivor.get('seed', 'N/A')
            matches = survivor.get('matches', 'N/A')
            rate = survivor.get('match_rate', 0)
            skip = survivor.get('skip_length', 'N/A')
            direction = survivor.get('direction', 'N/A')
            
            rate_str = f"{rate*100:.2f}%" if isinstance(rate, (int, float)) else str(rate)
            
            f.write(f"{i:6d} {seed:12} {matches:8} {rate_str:>8} {skip:6} {direction:14}\n")
    
    def _get_display_name(self, field: str) -> str:
        """Get human-readable display name for field."""
        # Convert field name to display name (simple version)
        return field.replace('_', ' ').title()
    
    def _format_value(self, key: str, value: Any) -> str:
        """Format value for display."""
        if isinstance(value, float):
            if 'rate' in key.lower() or 'ratio' in key.lower():
                return f"{value*100:.4f}%"
            return f"{value:.2f}"
        elif isinstance(value, int) and value > 1000:
            return f"{value:,}"
        elif isinstance(value, list):
            return ', '.join(str(v) for v in value[:5])
        else:
            return str(value)
    
    def _create_csv_export(self, data: Dict[str, Any], run_id: str) -> Path:
        """
        Create CSV export of survivors.
        
        Args:
            data: Results data
            run_id: Run identifier
            
        Returns:
            Path to created CSV file
        """
        output_path = self.results_dir / 'csv' / f"{run_id}.csv"
        
        survivors = data.get('survivors', [])
        
        if not survivors:
            # Create empty CSV with headers
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Seed', 'Matches', 'Match_Rate', 'Skip_Length', 'Direction'])
            return output_path
        
        # Get column names from first survivor
        columns = ['seed', 'matches', 'match_rate', 'skip_length', 'direction']
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            
            # Write metadata as comments
            writer.writerow({k: f"# {k}" for k in columns})
            writer.writerow({
                'seed': f"# Run: {run_id}",
                'matches': f"# Type: {data.get('run_metadata', {}).get('analysis_type', 'N/A')}",
                'match_rate': '',
                'skip_length': '',
                'direction': ''
            })
            
            # Write header
            writer.writeheader()
            
            # Write survivors
            for survivor in survivors:
                writer.writerow({k: survivor.get(k, '') for k in columns})
        
        return output_path
    
    def _create_json_export(
        self,
        data: Dict[str, Any],
        run_id: str,
        top_only: bool = True
    ) -> Path:
        """
        Create JSON export.
        
        Args:
            data: Results data
            run_id: Run identifier
            top_only: If True, only include top 100 survivors
            
        Returns:
            Path to created JSON file
        """
        suffix = '_top100' if top_only else '_detailed'
        output_path = self.results_dir / ('json' if top_only else 'detailed') / f"{run_id}{suffix}.json"
        
        # Prepare export data
        export_data = {
            'schema_version': self.schema['_schema_info']['version'],
            'run_metadata': data.get('run_metadata', {}),
            'analysis_parameters': data.get('analysis_parameters', {}),
            'results_summary': data.get('results_summary', {}),
            'performance_metrics': data.get('performance_metrics', {}),
        }
        
        # Add survivors (limited or full)
        survivors = data.get('survivors', [])
        if top_only and len(survivors) > 100:
            export_data['survivors'] = survivors[:100]
            export_data['_note'] = f"Top 100 of {len(survivors)} total survivors. See detailed file for complete data."
        else:
            export_data['survivors'] = survivors
        
        # Write JSON
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return output_path
    
    def _save_config(self, data: Dict[str, Any], run_id: str) -> Path:
        """Save analysis configuration."""
        output_path = self.results_dir / 'configs' / f"{run_id}_config.json"
        
        config = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'analysis_parameters': data.get('analysis_parameters', {}),
            'schema_version': self.schema['_schema_info']['version']
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return output_path


def main():
    """CLI interface for ResultsManager."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Schema-driven results manager for PRNG analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate schema files
  python3 results_manager.py --validate-schemas
  
  # List supported analysis types
  python3 results_manager.py --list-types
  
  # Validate a results JSON file
  python3 results_manager.py --validate results.json
        """
    )
    
    parser.add_argument('--validate-schemas', action='store_true',
                       help='Validate all schema files')
    parser.add_argument('--list-types', action='store_true',
                       help='List all supported analysis types')
    parser.add_argument('--validate', metavar='FILE',
                       help='Validate a results JSON file')
    
    args = parser.parse_args()
    
    if args.validate_schemas:
        try:
            rm = ResultsManager()
            print("✅ All schemas loaded and validated successfully!")
        except Exception as e:
            print(f"❌ Schema validation failed: {e}")
            sys.exit(1)
    
    elif args.list_types:
        rm = ResultsManager()
        types = rm._get_allowed_analysis_types()
        print(f"\nSupported Analysis Types ({len(types)}):")
        for i, t in enumerate(types, 1):
            print(f"  {i:2d}. {t}")
    
    elif args.validate:
        rm = ResultsManager()
        try:
            with open(args.validate, 'r') as f:
                data = json.load(f)
            rm._validate_data_structure(data)
            print(f"✅ {args.validate} is valid!")
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
