================================================================================
SCHEMAS DIRECTORY - DOCUMENTATION
================================================================================

Created: November 3, 2025
Purpose: Schema-driven configuration for PRNG analysis results system
Version: 1.0.2

================================================================================
FILES IN THIS DIRECTORY
================================================================================

1. results_schema_v1.json
   - Master data contract for all analysis results
   - Defines ALL possible fields, types, validation rules
   - Currently supports 24 analysis types
   - Version: 1.0.2
   
2. output_templates.json
   - Display format definitions for txt, csv, json outputs
   - Controls layout, sections, formatting
   - Version: 1.0.0
   
3. field_mappings.json
   - Maps internal names to display names
   - Controls field visibility and formatting
   - Version: 1.0.0

4. README.txt
   - This file - documentation for the schemas directory

================================================================================
HOW TO USE THESE SCHEMAS
================================================================================

FROM CODE:
----------
import json

# Load schema
schema = json.load(open('schemas/results_schema_v1.json'))

# Validate your data against schema
# Use schema['<section>']['fields'] to check field definitions

FROM COMMAND LINE:
------------------
# View schema structure
jq 'keys' schemas/results_schema_v1.json

# Check analysis types
jq '.run_metadata.fields.analysis_type.allowed_values' schemas/results_schema_v1.json

# See all ML metrics
jq '.ml_metrics.fields | keys' schemas/results_schema_v1.json

================================================================================
HOW TO EXTEND THE SCHEMAS
================================================================================

ADDING A NEW ANALYSIS TYPE:
----------------------------
1. Edit results_schema_v1.json
2. Add new type to: run_metadata.fields.analysis_type.allowed_values
3. Optionally add to: analysis_type_templates in output_templates.json
4. Optionally add to: analysis_type_specific_fields in field_mappings.json
5. Update version and modification_log

ADDING A NEW FIELD:
-------------------
1. Edit results_schema_v1.json
2. Add to appropriate section (run_metadata, analysis_parameters, etc.)
3. Specify: type, required, description, example
4. Add display name to field_mappings.json (optional)
5. Add to output template if needs special formatting (optional)
6. Update version and modification_log

CHANGING DISPLAY FORMAT:
------------------------
1. Edit output_templates.json
2. Modify summary_txt_template, csv_export_template, or json_export_template
3. Changes take effect immediately (no code changes needed)

================================================================================
REVISION HISTORY
================================================================================

Version 1.0.2 (2025-11-03):
- Added ML/RL analysis types and metrics
- Added 6 ML analysis types
- Added 13 ML-specific metric fields
- Total analysis types: 24

Version 1.0.1 (2025-11-03):
- Added 11 missing analysis types from codebase
- Total analysis types: 18

Version 1.0.0 (2025-11-03):
- Initial schema creation
- Core analysis types: bidirectional_sieve, forward_sieve, etc.
- Basic field definitions

================================================================================
MAINTENANCE
================================================================================

BACKUP BEFORE CHANGES:
cp results_schema_v1.json results_schema_v1.json.backup

VALIDATE JSON:
python3 -c "import json; json.load(open('results_schema_v1.json'))"

CHECK SCHEMA VERSION:
jq '._schema_info.version' results_schema_v1.json

VIEW MODIFICATION LOG:
jq '._schema_info.modification_log' results_schema_v1.json

================================================================================
