#!/usr/bin/env python3
"""
Models Package - Multi-Model Architecture v3.2.1
Team Beta Approved Implementation

v3.2.1 Changes:
- Added GlobalStateTracker (GPU-neutral module for 14 global features)
"""

from models.model_factory import (
    create_model,
    load_model,
    list_available_models,
    get_model_extension,
    get_model_defaults,
    save_model_with_sidecar,
    load_model_from_sidecar,
    AVAILABLE_MODELS,
    MODEL_EXTENSIONS,
    DEFAULT_MODELS_DIR,
)

from models.model_selector import (
    ModelSelector,
    SAFE_MODEL_ORDER,
)

from models.feature_schema import (
    get_feature_schema_from_data,
    get_feature_schema_with_hash,
    validate_feature_schema_hash,
)

from models.global_state_tracker import (
    GlobalStateTracker,
    GLOBAL_FEATURE_COUNT,
    GLOBAL_FEATURE_NAMES,
)

__all__ = [
    # model_factory
    'create_model',
    'load_model',
    'list_available_models',
    'get_model_extension',
    'get_model_defaults',
    'save_model_with_sidecar',
    'load_model_from_sidecar',
    'AVAILABLE_MODELS',
    'MODEL_EXTENSIONS',
    'DEFAULT_MODELS_DIR',
    # model_selector
    'ModelSelector',
    'SAFE_MODEL_ORDER',
    # feature_schema
    'get_feature_schema_from_data',
    'get_feature_schema_with_hash',
    'validate_feature_schema_hash',
    # global_state_tracker
    'GlobalStateTracker',
    'GLOBAL_FEATURE_COUNT',
    'GLOBAL_FEATURE_NAMES',
]

__version__ = "3.2.1"
